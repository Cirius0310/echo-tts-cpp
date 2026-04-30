#pragma once
// Echo-TTS C++ — Sampling (Euler ODE with CFG)
//
// Implements:
//   - sample_euler_cfg()     — standard Euler sampling with independent CFG
//   - sample_blockwise()     — blockwise generation with continuation support
//
// Both use a 3-pass CFG approach (one forward pass per condition variant)
// for maximum compatibility with GGML backends.

#include "echo_model.h"

#include <cstdint>
#include <string>
#include <vector>

// ────────────────────────────────────────────────────────────────────
// Sampler Configuration
// ────────────────────────────────────────────────────────────────────

struct EchoSamplerConfig {
    int      num_steps            = 40;
    float    cfg_scale_text       = 3.0f;
    float    cfg_scale_speaker    = 8.0f;
    float    cfg_min_t            = 0.5f;
    float    cfg_max_t            = 1.0f;
    float    truncation_factor    = 0.0f;   // 0 = disabled (no truncation)
    float    rescale_k            = 0.0f;   // 0 = disabled
    float    rescale_sigma        = 0.0f;   // 0 = disabled
    float    speaker_kv_scale     = 0.0f;   // 0 = disabled
    int      speaker_kv_max_layers = 0;     // 0 = all layers
    float    speaker_kv_min_t     = 0.0f;
    int      sequence_length      = 640;    // max 640 (training distribution)
    uint64_t rng_seed             = 0;
};

struct EchoBlockwiseConfig {
    EchoSamplerConfig    base;
    std::vector<int>     block_sizes;  // e.g. {128, 128, 64}
};

// ────────────────────────────────────────────────────────────────────
// Sampling result
// ────────────────────────────────────────────────────────────────────

struct EchoSamplerResult {
    std::vector<float> latent;   // (batch, seq_len, latent_size=80)
    int seq_len    = 0;
    int batch_size = 0;
};

// ────────────────────────────────────────────────────────────────────
// Sampling functions
// ────────────────────────────────────────────────────────────────────

// Standard Euler sampling with independent text + speaker CFG.
// speaker_latent: (1, seq_len, 80) — PCA-encoded speaker reference
// speaker_mask:   (1, seq_len)     — float32 (1.0=valid, 0.0=pad)
// text_ids:       (1, text_seq_len) — int32 token IDs
// text_mask:      (1, text_seq_len) — float32
EchoSamplerResult sample_euler_cfg(
    EchoModel & model,
    const EchoSamplerConfig & config,
    const float * speaker_latent,
    const float * speaker_mask,
    int speaker_seq_len,
    const int32_t * text_ids,
    const float * text_mask,
    int text_seq_len
);

// Standard Euler sampling with pre-computed speaker KV cache.
// kv_speaker: pre-computed speaker KV cache (shared across text chunks).
// speaker_mask, speaker_seq_len: still needed for unconditional CFG pass.
EchoSamplerResult sample_euler_cfg_with_speaker_kv(
    EchoModel & model,
    const EchoSamplerConfig & config,
    const float * speaker_mask,
    int speaker_seq_len,
    const int32_t * text_ids,
    const float * text_mask,
    int text_seq_len,
    const EchoKVCache & kv_speaker
);

// Blockwise Euler sampling with continuation support.
// continuation_latent: (1, cont_len, 80) — optional prefix latent (nullptr if none)
EchoSamplerResult sample_blockwise(
    EchoModel & model,
    const EchoBlockwiseConfig & config,
    const float * speaker_latent,
    const float * speaker_mask,
    int speaker_seq_len,
    const int32_t * text_ids,
    const float * text_mask,
    int text_seq_len,
    const float * continuation_latent = nullptr,
    int continuation_len = 0
);
