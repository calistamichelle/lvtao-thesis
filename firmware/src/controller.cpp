#include "controller.h"

#include <cmath>

#include "generated/controller_params.h"

void lvtao_features(const float window[][kAxes], WindowFeatures* out) {
    const float n = static_cast<float>(kWindow);
    float var[kAxes];
    float td[kAxes];
    for (int a = 0; a < kAxes; ++a) {
        float sum = 0.0f;
        for (int i = 0; i < kWindow; ++i) sum += window[i][a];
        const float mean = sum / n;

        float sq = 0.0f;
        for (int i = 0; i < kWindow; ++i) {
            const float d = window[i][a] - mean;
            sq += d * d;
        }
        var[a] = sq / n;

        float diff = 0.0f;
        for (int i = 1; i < kWindow; ++i) diff += fabsf(window[i][a] - window[i - 1][a]);
        td[a] = diff / static_cast<float>(kWindow - 1);
    }
    out->variance = (var[0] + var[1] + var[2]) / 3.0f;
    out->temporal_difference = (td[0] + td[1] + td[2]) / 3.0f;
    out->norm_variance = (logf(out->variance + kFeatureEps) - kMuLogVar) / kSdLogVar;
    out->norm_temporal_difference = (logf(out->temporal_difference + kFeatureEps) - kMuLogTd) / kSdLogTd;
}

static int level_of(float score, float t1, float t2) {
    return score < t1 ? 0 : (score < t2 ? 1 : 2);
}

void LvtaoController::reset() {
    level_ = kInitialLevel;
    dwell_ = 0;
    has_prev_ = false;
    prev_v_ = 0.0f;
    prev_d_ = 0.0f;
}

static float score_of(const WindowFeatures* f, bool has_prev, float prev_v, float prev_d) {
    const float v = f->norm_variance;
    const float d = f->norm_temporal_difference;
    // Change score (4): a transition shows up as a jump in either statistic between consecutive inference windows.
    return has_prev ? fmaxf(fabsf(v - prev_v), fabsf(d - prev_d)) : 0.0f;
}

int LvtaoController::step(WindowFeatures* f) {
    const float score = f->score = score_of(f, has_prev_, prev_v_, prev_d_);
    has_prev_ = true;
    prev_v_ = f->norm_variance;
    prev_d_ = f->norm_temporal_difference;

    // Move up immediately; move down only after kDwell decisions and past the hysteresis margin.
    const int up = level_of(score, kT1, kT2);
    if (up > level_) {
        level_ = up;
        dwell_ = 0;
    } else {
        const int down = level_of(score, kT1 - kHysteresis, kT2 - kHysteresis);
        if (down < level_ && dwell_ >= kDwell) {
            level_ = down;
            dwell_ = 0;
        } else {
            ++dwell_;
        }
    }
    return kStrides[level_];
}
