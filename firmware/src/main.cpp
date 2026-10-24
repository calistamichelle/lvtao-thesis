// LVTAO ESP32 test bench. Commands arrive as text lines over USB serial (UART0):
//   INFO                 model size, tensor arena, heap, stack, CPU clock
//   BENCH                time preprocessing / Invoke() / controller on windows stored in flash
//   REPLAY LVTAO         run the adaptive controller on samples streamed by python/host_stream.py
//   REPLAY FIXED <n>     same, with a fixed stride of n samples
// Replay frames from the host: 'R' = new segment, 'B' + uint16 count + count*3 float32 = samples,
// 'E' = end. The board answers each 'B' with one "D,..." line per inference and then "K".
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "controller.h"
#include "generated/bench_data.h"
#include "generated/model_data.h"
#include "generated/norm.h"

#if defined(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)
constexpr int kCpuMhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#elif defined(CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ)
constexpr int kCpuMhz = CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ;
#else
constexpr int kCpuMhz = 0;
#endif

namespace {

constexpr uart_port_t kUart = UART_NUM_0;
constexpr int kArenaSize = 48 * 1024;
constexpr int kBenchPasses = 8;
constexpr int kMaxBlock = 256;

alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ---------------------------------------------------------------- model ---

bool setup_model() {
    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERR,schema_version,%lu\n", static_cast<unsigned long>(model->version()));
        return false;
    }
    // Ops produced by the converter for Conv1D / MaxPooling1D / GAP / Dense / Softmax.
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddSqueeze();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERR,allocate_tensors\n");
        return false;
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    return true;
}

// Same float32 arithmetic and round-half-to-even as python/int8_model.py.
void fill_input(const float window[][kAxes]) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    int8_t* dst = input->data.int8;
    for (int i = 0; i < kWindow; ++i) {
        for (int a = 0; a < kAxes; ++a) {
            const float z = (window[i][a] - kNormMean[a]) / kNormStd[a];
            long q = lrintf(z / scale) + zero_point;
            dst[i * kAxes + a] = static_cast<int8_t>(std::min(127L, std::max(-128L, q)));
        }
    }
}

int run_model() {
    if (interpreter->Invoke() != kTfLiteOk) return -1;
    const int n = output->dims->data[output->dims->size - 1];
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (output->data.int8[i] > output->data.int8[best]) best = i;
    }
    return best;
}

// --------------------------------------------------------------- serial ---

bool read_exact(uint8_t* buf, int len, TickType_t timeout) {
    int got = 0;
    while (got < len) {
        const int n = uart_read_bytes(kUart, buf + got, len - got, timeout);
        if (n <= 0) return false;
        got += n;
    }
    return true;
}

bool read_line(char* buf, int cap) {
    int len = 0;
    while (true) {
        uint8_t c;
        if (uart_read_bytes(kUart, &c, 1, portMAX_DELAY) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') break;
        if (len < cap - 1) buf[len++] = static_cast<char>(c);
    }
    buf[len] = '\0';
    return len > 0;
}

// --------------------------------------------------------------- INFO -----

void cmd_info() {
    printf("INFO,model_bytes,%u\n", g_model_data_len);
    printf("INFO,arena_size_bytes,%d\n", kArenaSize);
    printf("INFO,arena_used_bytes,%u\n", static_cast<unsigned>(interpreter->arena_used_bytes()));
    printf("INFO,free_heap_bytes,%lu\n", static_cast<unsigned long>(esp_get_free_heap_size()));
    printf("INFO,min_free_heap_bytes,%u\n", static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT)));
    printf("INFO,main_stack_size_bytes,%d\n", CONFIG_ESP_MAIN_TASK_STACK_SIZE);
    printf("INFO,main_stack_unused_min_bytes,%u\n",
           static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr) * sizeof(StackType_t)));
    printf("INFO,cpu_mhz,%d\n", kCpuMhz);
    printf("INFO,input_scale,%.9g\n", input->params.scale);
    printf("INFO,input_zero_point,%d\n", static_cast<int>(input->params.zero_point));
    printf("INFO_END\n");
    fflush(stdout);
}

// --------------------------------------------------------------- BENCH ----

struct Stats {
    float mean, median, p95, max;
};

Stats summarize(uint32_t* v, int n) {
    std::sort(v, v + n);
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += v[i];
    return {static_cast<float>(sum / n), static_cast<float>(v[n / 2]),
            static_cast<float>(v[std::min(n - 1, (n * 95) / 100)]), static_cast<float>(v[n - 1])};
}

void cmd_bench() {
    constexpr int kRuns = kBenchWindows * kBenchPasses;
    static uint32_t pre_us[kRuns], invoke_us[kRuns], ctrl_us[kRuns];
    const auto* windows = reinterpret_cast<const float(*)[kWindow][kAxes]>(g_bench_raw);
    LvtaoController ctrl;
    WindowFeatures f;
    volatile int sink = 0;

    printf("BENCH_BEGIN,%d,%d\n", kBenchWindows, kBenchPasses);
    for (int p = 0; p < kBenchPasses; ++p) {
        for (int w = 0; w < kBenchWindows; ++w) {
            const int64_t t0 = esp_timer_get_time();
            fill_input(windows[w]);
            const int64_t t1 = esp_timer_get_time();
            const int pred = run_model();
            const int64_t t2 = esp_timer_get_time();
            lvtao_features(windows[w], &f);
            sink += ctrl.step(&f);
            const int64_t t3 = esp_timer_get_time();

            const int r = p * kBenchWindows + w;
            pre_us[r] = t1 - t0;
            invoke_us[r] = t2 - t1;
            ctrl_us[r] = t3 - t2;
            if (p == 0) printf("PRED,%d,%d\n", w, pred);
            if ((w & 7) == 0) vTaskDelay(1);  // outside the timed region
        }
    }
    const char* names[] = {"pre_us", "invoke_us", "ctrl_us"};
    uint32_t* series[] = {pre_us, invoke_us, ctrl_us};
    for (int i = 0; i < 3; ++i) {
        const Stats s = summarize(series[i], kRuns);
        printf("STAT,%s,%.2f,%.2f,%.2f,%.2f\n", names[i], s.mean, s.median, s.p95, s.max);
    }
    printf("BENCH_END\n");
    fflush(stdout);
    (void)sink;
}

// -------------------------------------------------------------- REPLAY ----

void cmd_replay(bool adaptive, int fixed_stride) {
    static float ring[kWindow][kAxes];
    static float window[kWindow][kAxes];
    static float block[kMaxBlock][kAxes];
    LvtaoController ctrl;
    WindowFeatures f = {};

    int head = 0;           // next write position in ring
    int seg_samples = 0;    // samples since the last segment reset
    int next_due = kWindow; // seg_samples value at which the next inference runs
    int selected = 0;       // stride that led to the current window
    uint32_t global = 0, inferences = 0;
    uint64_t push_total = 0, pre_total = 0, invoke_total = 0, ctrl_total = 0;
    uint32_t max_sample_work = 0;

    printf("READY\n");
    fflush(stdout);
    while (true) {
        uint8_t type;
        if (!read_exact(&type, 1, pdMS_TO_TICKS(30000))) {
            printf("ERR,replay_timeout\n");
            return;
        }
        if (type == 'R') {
            head = 0;
            seg_samples = 0;
            next_due = kWindow;
            selected = 0;
            ctrl.reset();
        } else if (type == 'B') {
            uint16_t count;
            if (!read_exact(reinterpret_cast<uint8_t*>(&count), 2, pdMS_TO_TICKS(1000)) || count > kMaxBlock ||
                !read_exact(reinterpret_cast<uint8_t*>(block), count * kAxes * sizeof(float), pdMS_TO_TICKS(1000))) {
                printf("ERR,bad_block\n");
                return;
            }
            for (int s = 0; s < count; ++s) {
                const int64_t t0 = esp_timer_get_time();
                memcpy(ring[head], block[s], sizeof(ring[head]));
                head = (head + 1) % kWindow;
                ++seg_samples;
                ++global;
                const int64_t t1 = esp_timer_get_time();
                uint32_t work = t1 - t0;
                push_total += work;

                if (seg_samples == next_due) {
                    const int64_t a = esp_timer_get_time();
                    for (int i = 0; i < kWindow; ++i) memcpy(window[i], ring[(head + i) % kWindow], sizeof(window[i]));
                    fill_input(window);
                    const int64_t b = esp_timer_get_time();
                    const int pred = run_model();
                    const int64_t c = esp_timer_get_time();
                    int stride = fixed_stride;
                    if (adaptive) {
                        lvtao_features(window, &f);
                        stride = ctrl.step(&f);
                    }
                    const int64_t d = esp_timer_get_time();

                    pre_total += b - a;
                    invoke_total += c - b;
                    ctrl_total += d - c;
                    work += d - a;
                    printf("D,%lu,%lu,%lu,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%lld,%lld,%lld\n",
                           static_cast<unsigned long>(inferences), static_cast<unsigned long>(global - kWindow),
                           static_cast<unsigned long>(global), selected, stride, pred,
                           f.variance, f.temporal_difference, f.norm_variance, f.norm_temporal_difference, f.score,
                           adaptive ? ctrl.level() : -1, b - a, c - b, d - c);
                    ++inferences;
                    next_due += stride;
                    selected = stride;
                }
                max_sample_work = std::max(max_sample_work, work);
            }
            printf("K\n");
            fflush(stdout);
        } else if (type == 'E') {
            printf("SUMMARY,%lu,%lu,%llu,%llu,%llu,%llu,%lu\n", static_cast<unsigned long>(global),
                   static_cast<unsigned long>(inferences), push_total, pre_total, invoke_total, ctrl_total,
                   static_cast<unsigned long>(max_sample_work));
            printf("DONE\n");
            fflush(stdout);
            return;
        }
    }
}

}  // namespace

extern "C" void app_main(void) {
    uart_driver_install(kUart, 8192, 0, 0, nullptr, 0);
    if (!setup_model()) {
        while (true) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    printf("LVTAO_FW READY\n");
    fflush(stdout);

    char line[64];
    while (true) {
        if (!read_line(line, sizeof(line))) continue;
        if (strcmp(line, "INFO") == 0) {
            cmd_info();
        } else if (strcmp(line, "BENCH") == 0) {
            cmd_bench();
        } else if (strcmp(line, "REPLAY LVTAO") == 0) {
            cmd_replay(true, 0);
        } else if (strncmp(line, "REPLAY FIXED ", 13) == 0) {
            const int stride = atoi(line + 13);
            if (stride > 0 && stride <= kWindow) {
                cmd_replay(false, stride);
            } else {
                printf("ERR,bad_stride\n");
            }
        } else {
            printf("ERR,unknown_command,%s\n", line);
        }
        fflush(stdout);
    }
}
