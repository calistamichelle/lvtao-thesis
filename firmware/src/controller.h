// LVTAO adaptive-stride controller. Mirrors python/policies.py line for line.
#pragma once

constexpr int kWindow = 128;
constexpr int kAxes = 3;

struct WindowFeatures {
    float variance;
    float temporal_difference;
    float norm_variance;
    float norm_temporal_difference;
    float score;  // filled in by LvtaoController::step
};

// Features of one raw window (oldest sample first), in g.
void lvtao_features(const float window[][kAxes], WindowFeatures* out);

class LvtaoController {
public:
    LvtaoController() { reset(); }
    void reset();
    // Scores the window (sets f->score) and returns the number of new samples
    // to wait for before the next inference.
    int step(WindowFeatures* f);
    int level() const { return level_; }

private:
    int level_;
    int dwell_;
    bool has_prev_;
    float prev_v_;
    float prev_d_;
};
