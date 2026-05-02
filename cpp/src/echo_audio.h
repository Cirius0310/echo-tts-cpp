#pragma once
// Echo-TTS C++ — Audio I/O
//
// WAV reading/writing using dr_wav.h (vendored in extern/).
// Includes a simple linear-interpolation resampler and volume normalization.

#include <string>
#include <vector>
#include <cstdint>

enum class NormalizeMode { None, Peak, RMS };

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

// Volume normalization for output audio.
// Peak mode: scale so max absolute sample equals target_peak (default 0.89 ≈ -1 dBFS).
void normalize_peak(float * audio, int num_samples, float target_peak = 0.89f);

// RMS mode: scale so RMS level equals target_rms (default 0.12 ≈ -18.4 dBFS).
// Includes a soft ceiling: if any sample exceeds ±1.0 after scaling, gain is reduced.
void normalize_rms(float * audio, int num_samples, float target_rms = 0.12f);

// Dispatch to the appropriate normalization function.
void normalize_audio_ex(NormalizeMode mode, float * audio, int num_samples, float target = 0.0f);

// Resample audio using linear interpolation.
// Returns resampled audio.
std::vector<float> resample_linear(
    const float * audio,
    int num_samples,
    int from_sr,
    int to_sr
);
