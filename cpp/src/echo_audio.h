#pragma once
// Echo-TTS C++ — Audio I/O
//
// WAV reading/writing using dr_wav.h (vendored in extern/).
// Includes a simple linear-interpolation resampler for v1.

#include <string>
#include <vector>
#include <cstdint>

// Load a WAV file, convert to mono float32, resample to target_sr.
// Returns empty vector on failure.
std::vector<float> load_wav(
    const std::string & path,
    int target_sr = 44100,
    float max_duration_sec = 300.0f
);

// Save mono float32 audio to a WAV file.
bool save_wav(
    const std::string & path,
    const float * audio,
    int num_samples,
    int sample_rate = 44100
);

// Normalize audio: divide by max(max_abs, 1.0).
void normalize_audio(float * audio, int num_samples);

// Resample audio using linear interpolation.
// Returns resampled audio.
std::vector<float> resample_linear(
    const float * audio,
    int num_samples,
    int from_sr,
    int to_sr
);
