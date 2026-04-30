#pragma once
// Echo-TTS C++ — Post-processing (flattening detection + audio cropping)

#include <vector>

// AE downsample factor: S1-DAC produces 1 latent frame per 2048 audio samples
constexpr int AE_DOWNSAMPLE_FACTOR = 2048;

// Find the point where the latent flattens to near-zero (end of generation).
// Returns the frame index where generation ends.
//
// latent: (time_steps, latent_size) row-major
int find_flattening_point(
    const float * latent,
    int time_steps,
    int latent_size,
    float target_value = 0.0f,
    int   window_size  = 20,
    float std_threshold = 0.05f,
    float mean_threshold = 0.1f
);

// Crop audio to the flattening point of the latent.
// Returns the number of audio samples to keep.
int crop_length_from_latent(
    const float * latent,
    int time_steps,
    int latent_size
);
