// Echo-TTS C++ — Post-processing Implementation

#include "echo_postprocess.h"
#include <cmath>
#include <algorithm>

int find_flattening_point(
    const float * latent,
    int time_steps,
    int latent_size,
    float target_value,
    int   window_size,
    float std_threshold,
    float mean_threshold
) {
    // Mirrors Python: slide a window of `window_size` frames and check if
    // the window has std < std_threshold and |mean - target| < mean_threshold.
    // Pad with zeros at the end (like the Python version).

    int padded_len = time_steps + window_size;

    for (int i = 0; i < padded_len - window_size; i++) {
        // Compute mean and std of the window across all latent dims
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;

        for (int w = 0; w < window_size; w++) {
            int frame = i + w;
            for (int d = 0; d < latent_size; d++) {
                float val;
                if (frame < time_steps) {
                    val = latent[frame * latent_size + d];
                } else {
                    val = 0.0f;  // zero padding
                }
                sum += val;
                sum_sq += val * val;
                count++;
            }
        }

        float mean = sum / count;
        float variance = sum_sq / count - mean * mean;
        float std_dev = std::sqrt(std::max(0.0f, variance));

        if (std_dev < std_threshold && std::abs(mean - target_value) < mean_threshold) {
            return i;
        }
    }

    return time_steps;
}

int crop_length_from_latent(
    const float * latent,
    int time_steps,
    int latent_size
) {
    int flat_point = find_flattening_point(latent, time_steps, latent_size);
    return flat_point * AE_DOWNSAMPLE_FACTOR;
}
