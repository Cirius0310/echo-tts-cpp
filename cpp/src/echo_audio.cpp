// Echo-TTS C++ — Audio I/O Implementation

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "echo_audio.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ── Resample ────────────────────────────────────────────────────────

std::vector<float> resample_linear(
    const float * audio,
    int num_samples,
    int from_sr,
    int to_sr
) {
    if (from_sr == to_sr || num_samples == 0) {
        return std::vector<float>(audio, audio + num_samples);
    }

    double ratio = static_cast<double>(to_sr) / from_sr;
    int out_samples = static_cast<int>(std::ceil(num_samples * ratio));
    std::vector<float> out(out_samples);

    for (int i = 0; i < out_samples; i++) {
        double src_idx = i / ratio;
        int idx0 = static_cast<int>(src_idx);
        int idx1 = std::min(idx0 + 1, num_samples - 1);
        double frac = src_idx - idx0;

        out[i] = static_cast<float>(audio[idx0] * (1.0 - frac) + audio[idx1] * frac);
    }

    return out;
}

// ── Normalize ───────────────────────────────────────────────────────

void normalize_audio(float * audio, int num_samples) {
    float max_abs = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        max_abs = std::max(max_abs, std::abs(audio[i]));
    }
    float divisor = std::max(max_abs, 1.0f);
    for (int i = 0; i < num_samples; i++) {
        audio[i] /= divisor;
    }
}

// ── Volume normalization (output) ────────────────────────────────────

void normalize_peak(float * audio, int num_samples, float target_peak) {
    if (num_samples <= 0) return;

    float max_abs = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        max_abs = std::max(max_abs, std::abs(audio[i]));
    }

    if (max_abs < 1e-8f) return;  // silence, skip

    float gain = target_peak / max_abs;
    for (int i = 0; i < num_samples; i++) {
        audio[i] *= gain;
    }
}

void normalize_rms(float * audio, int num_samples, float target_rms) {
    if (num_samples <= 0) return;

    double sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum_sq += (double)audio[i] * (double)audio[i];
    }

    float rms = (float)std::sqrt(sum_sq / num_samples);
    if (rms < 1e-8f) return;  // silence, skip

    float gain = target_rms / rms;

    // Soft ceiling: if any sample would exceed ±1.0, clamp gain
    float peak_after = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float v = audio[i] * gain;
        if (v > peak_after)  peak_after = v;
        if (-v > peak_after) peak_after = -v;
    }

    if (peak_after > 1.0f) {
        gain /= peak_after;  // reduce gain so peak stays at 1.0
    }

    for (int i = 0; i < num_samples; i++) {
        audio[i] *= gain;
    }
}

void normalize_audio_ex(NormalizeMode mode, float * audio, int num_samples, float target) {
    switch (mode) {
        case NormalizeMode::Peak:
            normalize_peak(audio, num_samples, target > 0.0f ? target : 0.89f);
            break;
        case NormalizeMode::RMS:
            normalize_rms(audio, num_samples, target > 0.0f ? target : 0.12f);
            break;
        case NormalizeMode::None:
        default:
            break;
    }
}

// ── Load WAV ────────────────────────────────────────────────────────

std::vector<float> load_wav(
    const std::string & path,
    int target_sr,
    float max_duration_sec
) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "[echo_audio] Failed to open WAV file: %s\n", path.c_str());
        return {};
    }

    // Limit to max duration
    uint64_t max_frames = static_cast<uint64_t>(max_duration_sec * wav.sampleRate);
    uint64_t frames_to_read = std::min(static_cast<uint64_t>(wav.totalPCMFrameCount), max_frames);

    // Read all channels as interleaved float32
    std::vector<float> interleaved(frames_to_read * wav.channels);
    uint64_t frames_read = drwav_read_pcm_frames_f32(&wav, frames_to_read, interleaved.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        fprintf(stderr, "[echo_audio] No frames read from: %s\n", path.c_str());
        return {};
    }

    // Convert to mono by averaging channels
    std::vector<float> mono(frames_read);
    for (uint64_t i = 0; i < frames_read; i++) {
        float sum = 0.0f;
        for (uint32_t ch = 0; ch < wav.channels; ch++) {
            sum += interleaved[i * wav.channels + ch];
        }
        mono[i] = sum / wav.channels;
    }

    // Normalize
    normalize_audio(mono.data(), static_cast<int>(mono.size()));

    // Resample to target sample rate
    if (static_cast<int>(wav.sampleRate) != target_sr) {
        mono = resample_linear(mono.data(), static_cast<int>(mono.size()),
                               static_cast<int>(wav.sampleRate), target_sr);
    }

    return mono;
}

// ── Save WAV ────────────────────────────────────────────────────────

bool save_wav(
    const std::string & path,
    const float * audio,
    int num_samples,
    int sample_rate
) {
    drwav_data_format format = {};
    format.container   = drwav_container_riff;
    format.format      = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels    = 1;
    format.sampleRate  = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        fprintf(stderr, "[echo_audio] Failed to create WAV file: %s\n", path.c_str());
        return false;
    }

    drwav_uint64 written = drwav_write_pcm_frames(&wav, num_samples, audio);
    drwav_uninit(&wav);

    return written == static_cast<drwav_uint64>(num_samples);
}
