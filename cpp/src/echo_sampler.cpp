// Echo-TTS C++ — Sampler Implementation
//
// Euler ODE sampling with independent text + speaker CFG.
// Uses 3 separate forward passes for CFG (safe fallback for GGML backends).

#include "echo_sampler.h"
#include "echo_pca.h"

#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <vector>
#include <chrono>

// ────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────

// Temporal score rescaling (arXiv:2510.01184)
static void temporal_score_rescale(
    float * v_pred,        // (batch * seq * 80) — modified in-place
    const float * x_t,     // (batch * seq * 80)
    float t,
    float rescale_k,
    float rescale_sigma,
    int total_elements
) {
    if (t >= 1.0f) return;
    float snr = (1.0f - t) * (1.0f - t) / (t * t);
    float s2 = rescale_sigma * rescale_sigma;
    float ratio = (snr * s2 + 1.0f) / (snr * s2 / rescale_k + 1.0f);
    float inv_1mt = 1.0f / (1.0f - t);

    for (int i = 0; i < total_elements; i++) {
        float scaled = ratio * ((1.0f - t) * v_pred[i] + x_t[i]) - x_t[i];
        v_pred[i] = inv_1mt * scaled;
    }
}

// Scale KV cache values in-place by reading, scaling, and writing back
static void scale_kv_cache(
    EchoKVCache & cache,
    float scale,
    int max_layers  // 0 = all
) {
    int n_layers = (max_layers > 0 && max_layers < (int)cache.layers.size())
                   ? max_layers : (int)cache.layers.size();

    for (int i = 0; i < n_layers; i++) {
        auto & kv = cache.layers[i];

        size_t k_size = ggml_nelements(kv.k);
        size_t v_size = ggml_nelements(kv.v);

        std::vector<float> k_data(k_size);
        std::vector<float> v_data(v_size);

        ggml_backend_tensor_get(kv.k, k_data.data(), 0, k_size * sizeof(float));
        ggml_backend_tensor_get(kv.v, v_data.data(), 0, v_size * sizeof(float));

        for (size_t j = 0; j < k_size; j++) k_data[j] *= scale;
        for (size_t j = 0; j < v_size; j++) v_data[j] *= scale;

        ggml_backend_tensor_set(kv.k, k_data.data(), 0, k_size * sizeof(float));
        ggml_backend_tensor_set(kv.v, v_data.data(), 0, v_size * sizeof(float));
    }
}

// ────────────────────────────────────────────────────────────────────
// Standard Euler sampling with independent CFG
// ────────────────────────────────────────────────────────────────────

EchoSamplerResult sample_euler_cfg(
    EchoModel & model,
    const EchoSamplerConfig & config,
    const float * speaker_latent,
    const float * speaker_mask,
    int speaker_seq_len,
    const int32_t * text_ids,
    const float * text_mask,
    int text_seq_len
) {
    auto & hp = model.hparams();
    int seq_len = config.sequence_length;
    int batch_size = 1;
    int latent_dim = hp.latent_size;  // 80
    int total_latent = batch_size * seq_len * latent_dim;

    const float INIT_SCALE = 0.999f;

    auto t_start = std::chrono::high_resolution_clock::now();

    printf("[sampler] Computing KV caches...\n");

    // Compute text and speaker KV caches
    EchoKVCache kv_text = model.compute_text_kv_cache(text_ids, text_mask, text_seq_len, batch_size);
    EchoKVCache kv_speaker = model.compute_speaker_kv_cache(speaker_latent, speaker_seq_len, batch_size);

    // Optional speaker KV scaling
    if (config.speaker_kv_scale > 0.0f && config.speaker_kv_scale != 1.0f) {
        scale_kv_cache(kv_speaker, config.speaker_kv_scale, config.speaker_kv_max_layers);
    }

    // Create uncond masks (all zeros — prevents attending to anything)
    std::vector<float> text_mask_uncond(text_seq_len * batch_size, 0.0f);
    std::vector<float> speaker_mask_uncond(speaker_seq_len * batch_size, 0.0f);

    // Schedule: t_schedule = linspace(1, 0, num_steps+1) * INIT_SCALE
    std::vector<float> t_schedule(config.num_steps + 1);
    for (int i = 0; i <= config.num_steps; i++) {
        t_schedule[i] = (1.0f - (float)i / config.num_steps) * INIT_SCALE;
    }

    // Initialize x_t with Gaussian noise
    std::mt19937_64 rng(config.rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x_t(total_latent);
    for (int i = 0; i < total_latent; i++) {
        x_t[i] = normal(rng);
    }
    // Apply truncation
    if (config.truncation_factor > 0.0f && config.truncation_factor != 1.0f) {
        for (int i = 0; i < total_latent; i++) {
            x_t[i] *= config.truncation_factor;
        }
    }

    printf("[sampler] Starting Euler sampling: %d steps, seq_len=%d\n",
           config.num_steps, seq_len);

    // ── Euler loop ──
    for (int step = 0; step < config.num_steps; step++) {
        float t = t_schedule[step];
        float t_next = t_schedule[step + 1];
        float dt = t_next - t;

        bool has_cfg = (t >= config.cfg_min_t && t <= config.cfg_max_t);

        std::vector<float> v_pred;

        if (has_cfg) {
            // 3-pass CFG: compute v_cond, v_uncond_text, v_uncond_speaker

            // Pass 1: conditioned (full text mask + full speaker mask)
            std::vector<float> t_vec(batch_size, t);
            std::vector<float> v_cond = model.forward(
                x_t.data(), t_vec.data(), seq_len, batch_size,
                text_mask, text_seq_len,
                speaker_mask, speaker_seq_len,
                kv_text, kv_speaker
            );

            // Pass 2: uncond text (zero text mask + full speaker mask)
            std::vector<float> v_uncond_text = model.forward(
                x_t.data(), t_vec.data(), seq_len, batch_size,
                text_mask_uncond.data(), text_seq_len,
                speaker_mask, speaker_seq_len,
                kv_text, kv_speaker
            );

            // Pass 3: uncond speaker (full text mask + zero speaker mask)
            std::vector<float> v_uncond_speaker = model.forward(
                x_t.data(), t_vec.data(), seq_len, batch_size,
                text_mask, text_seq_len,
                speaker_mask_uncond.data(), speaker_seq_len,
                kv_text, kv_speaker
            );

            // CFG: v_pred = v_cond + w_text*(v_cond - v_uncond_text) + w_speaker*(v_cond - v_uncond_speaker)
            v_pred.resize(total_latent);
            for (int i = 0; i < total_latent; i++) {
                v_pred[i] = v_cond[i]
                    + config.cfg_scale_text    * (v_cond[i] - v_uncond_text[i])
                    + config.cfg_scale_speaker * (v_cond[i] - v_uncond_speaker[i]);
            }
        } else {
            // No CFG: single conditioned pass
            std::vector<float> t_vec(batch_size, t);
            v_pred = model.forward(
                x_t.data(), t_vec.data(), seq_len, batch_size,
                text_mask, text_seq_len,
                speaker_mask, speaker_seq_len,
                kv_text, kv_speaker
            );
        }

        // Optional temporal score rescaling
        if (config.rescale_k > 0.0f && config.rescale_sigma > 0.0f) {
            temporal_score_rescale(v_pred.data(), x_t.data(), t,
                                  config.rescale_k, config.rescale_sigma, total_latent);
        }

        // Optional KV speaker un-scaling at threshold
        if (config.speaker_kv_scale > 0.0f && config.speaker_kv_scale != 1.0f
            && t_next < config.speaker_kv_min_t && t >= config.speaker_kv_min_t) {
            scale_kv_cache(kv_speaker, 1.0f / config.speaker_kv_scale, config.speaker_kv_max_layers);
        }

        // Euler step: x_t += v_pred * dt
        for (int i = 0; i < total_latent; i++) {
            x_t[i] += v_pred[i] * dt;
        }

        if ((step + 1) % 10 == 0 || step == 0) {
            printf("[sampler] Step %d/%d  t=%.3f\n", step + 1, config.num_steps, t);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[sampler] Sampling complete in %.1f seconds\n", elapsed);

    EchoSamplerResult result;
    result.latent = std::move(x_t);
    result.seq_len = seq_len;
    result.batch_size = batch_size;
    return result;
}

// ────────────────────────────────────────────────────────────────────
// Blockwise Euler sampling
// ────────────────────────────────────────────────────────────────────

EchoSamplerResult sample_blockwise(
    EchoModel & model,
    const EchoBlockwiseConfig & config,
    const float * speaker_latent,
    const float * speaker_mask,
    int speaker_seq_len,
    const int32_t * text_ids,
    const float * text_mask,
    int text_seq_len,
    const float * continuation_latent,
    int continuation_len
) {
    auto & hp = model.hparams();
    auto & cfg = config.base;
    int batch_size = 1;
    int latent_dim = hp.latent_size;  // 80

    const float INIT_SCALE = 0.999f;

    auto t_start = std::chrono::high_resolution_clock::now();

    printf("[sampler] Blockwise sampling: %zu blocks\n", config.block_sizes.size());

    // Compute text and speaker KV caches
    EchoKVCache kv_text = model.compute_text_kv_cache(text_ids, text_mask, text_seq_len, batch_size);
    EchoKVCache kv_speaker = model.compute_speaker_kv_cache(speaker_latent, speaker_seq_len, batch_size);

    // Create uncond masks
    std::vector<float> text_mask_uncond(text_seq_len * batch_size, 0.0f);
    std::vector<float> speaker_mask_uncond(speaker_seq_len * batch_size, 0.0f);

    // Schedule
    std::vector<float> t_schedule(cfg.num_steps + 1);
    for (int i = 0; i <= cfg.num_steps; i++) {
        t_schedule[i] = (1.0f - (float)i / cfg.num_steps) * INIT_SCALE;
    }

    // Total sequence: continuation + all blocks
    int total_block_len = 0;
    for (int bs : config.block_sizes) total_block_len += bs;
    int total_seq = continuation_len + total_block_len;

    // Prefix latent buffer (holds all generated latents)
    std::vector<float> prefix_latent(total_seq * latent_dim, 0.0f);
    int start_pos = 0;

    // Copy continuation latent if provided
    if (continuation_latent && continuation_len > 0) {
        std::copy(continuation_latent,
                  continuation_latent + continuation_len * latent_dim,
                  prefix_latent.data());
        start_pos = continuation_len;
    }

    // RNG
    std::mt19937_64 rng(cfg.rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // ── Block loop ──
    for (size_t bi = 0; bi < config.block_sizes.size(); bi++) {
        int block_size = config.block_sizes[bi];
        printf("[sampler] Block %zu/%zu: size=%d, start_pos=%d\n",
               bi + 1, config.block_sizes.size(), block_size, start_pos);

        // Scale speaker KV (re-apply at start of each block)
        if (cfg.speaker_kv_scale > 0.0f && cfg.speaker_kv_scale != 1.0f) {
            scale_kv_cache(kv_speaker, cfg.speaker_kv_scale, cfg.speaker_kv_max_layers);
        }

        // Compute latent KV cache from all prefix latent so far
        EchoKVCache kv_latent;
        if (start_pos > 0) {
            kv_latent = model.compute_latent_kv_cache(
                prefix_latent.data(), start_pos, batch_size
            );
        }

        // Initialize block noise
        int block_latent_size = block_size * latent_dim;
        std::vector<float> x_t(block_latent_size);
        for (int i = 0; i < block_latent_size; i++) {
            x_t[i] = normal(rng);
        }
        if (cfg.truncation_factor > 0.0f && cfg.truncation_factor != 1.0f) {
            for (int i = 0; i < block_latent_size; i++) {
                x_t[i] *= cfg.truncation_factor;
            }
        }

        // ── Euler loop for this block ──
        for (int step = 0; step < cfg.num_steps; step++) {
            float t = t_schedule[step];
            float t_next = t_schedule[step + 1];
            float dt = t_next - t;

            bool has_cfg = (t >= cfg.cfg_min_t && t <= cfg.cfg_max_t);

            std::vector<float> v_pred;
            EchoKVCache * lat_ptr = (start_pos > 0) ? &kv_latent : nullptr;

            if (has_cfg) {
                std::vector<float> t_vec(batch_size, t);

                std::vector<float> v_cond = model.forward(
                    x_t.data(), t_vec.data(), block_size, batch_size,
                    text_mask, text_seq_len,
                    speaker_mask, speaker_seq_len,
                    kv_text, kv_speaker, start_pos, lat_ptr
                );
                std::vector<float> v_uncond_text = model.forward(
                    x_t.data(), t_vec.data(), block_size, batch_size,
                    text_mask_uncond.data(), text_seq_len,
                    speaker_mask, speaker_seq_len,
                    kv_text, kv_speaker, start_pos, lat_ptr
                );
                std::vector<float> v_uncond_speaker = model.forward(
                    x_t.data(), t_vec.data(), block_size, batch_size,
                    text_mask, text_seq_len,
                    speaker_mask_uncond.data(), speaker_seq_len,
                    kv_text, kv_speaker, start_pos, lat_ptr
                );

                v_pred.resize(block_latent_size);
                for (int i = 0; i < block_latent_size; i++) {
                    v_pred[i] = v_cond[i]
                        + cfg.cfg_scale_text    * (v_cond[i] - v_uncond_text[i])
                        + cfg.cfg_scale_speaker * (v_cond[i] - v_uncond_speaker[i]);
                }
            } else {
                std::vector<float> t_vec(batch_size, t);
                v_pred = model.forward(
                    x_t.data(), t_vec.data(), block_size, batch_size,
                    text_mask, text_seq_len,
                    speaker_mask, speaker_seq_len,
                    kv_text, kv_speaker, start_pos, lat_ptr
                );
            }

            // Temporal score rescaling
            if (cfg.rescale_k > 0.0f && cfg.rescale_sigma > 0.0f) {
                temporal_score_rescale(v_pred.data(), x_t.data(), t,
                                      cfg.rescale_k, cfg.rescale_sigma, block_latent_size);
            }

            // KV speaker un-scaling
            if (cfg.speaker_kv_scale > 0.0f && cfg.speaker_kv_scale != 1.0f
                && t_next < cfg.speaker_kv_min_t && t >= cfg.speaker_kv_min_t) {
                scale_kv_cache(kv_speaker, 1.0f / cfg.speaker_kv_scale, cfg.speaker_kv_max_layers);
            }

            // Euler step
            for (int i = 0; i < block_latent_size; i++) {
                x_t[i] += v_pred[i] * dt;
            }
        }

        // Copy block result into prefix
        std::copy(x_t.begin(), x_t.end(),
                  prefix_latent.begin() + start_pos * latent_dim);
        start_pos += block_size;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[sampler] Blockwise sampling complete in %.1f seconds\n", elapsed);

    EchoSamplerResult result;
    result.latent = std::move(prefix_latent);
    result.seq_len = total_seq;
    result.batch_size = batch_size;
    return result;
}
