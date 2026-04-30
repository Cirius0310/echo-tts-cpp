#pragma once
// Echo-TTS C++ Inference Engine — Model Definitions
//
// Implements the EchoDiT architecture:
//   - TextEncoder (14-layer bidirectional transformer)
//   - SpeakerEncoder (14-layer causal transformer, patch_size=4)
//   - LatentEncoder (same architecture, for blockwise mode)
//   - Diffusion Decoder (24-layer DiT with joint attention + AdaLN)
//
// Uses GGML for tensor computation and GGUF for weight storage.
// The model manages all GGML backend state internally and exposes
// a clean API that takes/returns raw float arrays.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// ────────────────────────────────────────────────────────────────────
// Hyperparameters
// ────────────────────────────────────────────────────────────────────

struct EchoHParams {
    // Decoder (DiT)
    int32_t latent_size         = 80;
    int32_t model_size          = 2048;
    int32_t num_layers          = 24;
    int32_t num_heads           = 16;
    int32_t intermediate_size   = 5888;
    float   norm_eps            = 1e-5f;

    // Text encoder
    int32_t text_vocab_size     = 256;
    int32_t text_model_size     = 1280;
    int32_t text_num_layers     = 14;
    int32_t text_num_heads      = 10;
    int32_t text_intermediate_size = 3328;

    // Speaker encoder
    int32_t speaker_patch_size  = 4;
    int32_t speaker_model_size  = 1280;
    int32_t speaker_num_layers  = 14;
    int32_t speaker_num_heads   = 10;
    int32_t speaker_intermediate_size = 3328;

    // Conditioning
    int32_t timestep_embed_size = 512;
    int32_t adaln_rank          = 256;

    // Derived
    int32_t head_dim() const { return model_size / num_heads; }
    int32_t text_head_dim() const { return text_model_size / text_num_heads; }
    int32_t speaker_head_dim() const { return speaker_model_size / speaker_num_heads; }

    bool include_blockwise = true;
};

// ────────────────────────────────────────────────────────────────────
// PCA State (stored as CPU floats for use by echo_pca.h)
// ────────────────────────────────────────────────────────────────────

struct EchoPCAState {
    std::vector<float> components;  // (80 * 1024) row-major
    std::vector<float> mean;        // (1024,)
    float latent_scale = 1.0f;
};

// ────────────────────────────────────────────────────────────────────
// KV Cache — stored on-device as backend buffers
// ────────────────────────────────────────────────────────────────────

struct EchoKVPair {
    struct ggml_tensor * k;  // (head_dim, seq_len, num_heads, batch)
    struct ggml_tensor * v;  // (head_dim, seq_len, num_heads, batch)
};

struct EchoKVCache {
    std::vector<EchoKVPair>       layers;
    struct ggml_context *         ctx    = nullptr;   // owns tensor metadata
    ggml_backend_buffer_t         buffer = nullptr;   // owns tensor data on device
    int seq_len    = 0;
    int batch_size = 0;

    void free();
    ~EchoKVCache() { free(); }

    // Move-only
    EchoKVCache() = default;
    EchoKVCache(EchoKVCache && o) noexcept;
    EchoKVCache & operator=(EchoKVCache && o) noexcept;
    EchoKVCache(const EchoKVCache &) = delete;
    EchoKVCache & operator=(const EchoKVCache &) = delete;
};

// ────────────────────────────────────────────────────────────────────
// Layer Weight Structures (pointers into the weight context)
// ────────────────────────────────────────────────────────────────────

struct SelfAttentionWeights {
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * gate;
    struct ggml_tensor * q_norm;   // (num_heads, head_dim)
    struct ggml_tensor * k_norm;
};

struct JointAttentionWeights {
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wk_text;
    struct ggml_tensor * wv_text;
    struct ggml_tensor * wk_speaker;
    struct ggml_tensor * wv_speaker;
    struct ggml_tensor * wk_latent;
    struct ggml_tensor * wv_latent;
    struct ggml_tensor * q_norm;
    struct ggml_tensor * k_norm;
    struct ggml_tensor * gate;
    struct ggml_tensor * wo;
};

struct MLPWeights {
    struct ggml_tensor * w1;  // (model_size, intermediate_size)
    struct ggml_tensor * w3;  // (model_size, intermediate_size)
    struct ggml_tensor * w2;  // (intermediate_size, model_size)
};

struct LowRankAdaLNWeights {
    struct ggml_tensor * shift_down;
    struct ggml_tensor * scale_down;
    struct ggml_tensor * gate_down;
    struct ggml_tensor * shift_up;
    struct ggml_tensor * scale_up;
    struct ggml_tensor * gate_up;
    struct ggml_tensor * shift_up_bias;
    struct ggml_tensor * scale_up_bias;
    struct ggml_tensor * gate_up_bias;
};

struct EncoderBlockWeights {
    SelfAttentionWeights attention;
    MLPWeights mlp;
    struct ggml_tensor * attention_norm;
    struct ggml_tensor * mlp_norm;
};

struct DecoderBlockWeights {
    JointAttentionWeights attention;
    MLPWeights mlp;
    LowRankAdaLNWeights attention_adaln;
    LowRankAdaLNWeights mlp_adaln;
};

// ────────────────────────────────────────────────────────────────────
// Full Model Weight Structure
// ────────────────────────────────────────────────────────────────────

struct EchoModelWeights {
    // Text encoder
    struct ggml_tensor * text_embedding;
    std::vector<EncoderBlockWeights> text_blocks;

    // Speaker encoder
    struct ggml_tensor * speaker_in_proj;
    struct ggml_tensor * speaker_in_proj_bias;
    std::vector<EncoderBlockWeights> speaker_blocks;

    // Latent encoder (blockwise)
    struct ggml_tensor * latent_in_proj;
    struct ggml_tensor * latent_in_proj_bias;
    std::vector<EncoderBlockWeights> latent_blocks;

    // Post-encoder norms
    struct ggml_tensor * text_norm;
    struct ggml_tensor * speaker_norm;
    struct ggml_tensor * latent_norm;

    // Conditioning: timestep_embed → 3 Linears with SiLU
    struct ggml_tensor * cond_linear0;   // (timestep_embed_size, model_size)
    struct ggml_tensor * cond_linear1;   // (model_size, model_size)
    struct ggml_tensor * cond_linear2;   // (model_size, model_size*3)

    // Input projection
    struct ggml_tensor * in_proj;
    struct ggml_tensor * in_proj_bias;

    // Decoder blocks
    std::vector<DecoderBlockWeights> decoder_blocks;

    // Output
    struct ggml_tensor * out_norm;
    struct ggml_tensor * out_proj;
    struct ggml_tensor * out_proj_bias;
};

// ────────────────────────────────────────────────────────────────────
// Model class
// ────────────────────────────────────────────────────────────────────

class EchoModel {
public:
    EchoModel();
    ~EchoModel();

    // Load model from GGUF file.
    bool load(const std::string & path);

    // ── Encoder methods (precompute KV caches) ──

    // Run text encoder → KV cache for all decoder layers.
    // input_ids: (batch_size, seq_len) int32
    // mask:      (batch_size, seq_len) float32 (1.0 = attend, 0.0 = mask)
    EchoKVCache compute_text_kv_cache(
        const int32_t * input_ids,
        const float * mask,
        int seq_len,
        int batch_size
    );

    // Run speaker encoder → KV cache for all decoder layers.
    // speaker_latent: (batch_size, seq_len, latent_size=80) float32
    EchoKVCache compute_speaker_kv_cache(
        const float * speaker_latent,
        int seq_len,
        int batch_size
    );

    // Run latent encoder → KV cache for all decoder layers (blockwise).
    EchoKVCache compute_latent_kv_cache(
        const float * prefix_latent,
        int seq_len,
        int batch_size
    );

    // ── Decoder forward pass ──

    // x:        (batch_size, seq_len, latent_size) — noised latents
    // t:        (batch_size,) — timesteps
    // text_mask, speaker_mask: (batch_size, respective_seq_len) float32
    // Returns:  (batch_size, seq_len, latent_size) — velocity prediction
    std::vector<float> forward(
        const float * x,
        const float * t,
        int seq_len,
        int batch_size,
        const float * text_mask,
        int text_seq_len,
        const float * speaker_mask,
        int speaker_seq_len,
        const EchoKVCache & kv_text,
        const EchoKVCache & kv_speaker,
        int start_pos = 0,
        const EchoKVCache * kv_latent = nullptr
    );

    // Access
    const EchoHParams &  hparams()   const { return hparams_; }
    const EchoPCAState & pca_state() const { return pca_state_; }
    ggml_backend_t       backend()   const { return backend_; }

private:
    EchoHParams hparams_;
    EchoPCAState pca_state_;
    EchoModelWeights weights_;

    // GGML state
    struct ggml_context *   weight_ctx_  = nullptr;
    struct gguf_context *   gguf_ctx_    = nullptr;
    ggml_backend_t          backend_     = nullptr;
    ggml_backend_t          cpu_backend_ = nullptr;
    ggml_backend_buffer_t   weight_buf_  = nullptr;
    ggml_backend_sched_t    sched_       = nullptr;

    // CPU copy for the small timestep conditioning MLP. The final conditioning
    // projection is a large matrix-vector multiply that is unreliable on the
    // current GGML CUDA path when stored as F16.
    std::vector<float> cond_w0_cpu_;
    std::vector<float> cond_w1_cpu_;
    std::vector<float> cond_w2_cpu_;

    // ── Graph-building helpers ──

    struct ggml_tensor * build_rms_norm(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * weight, float eps);

    struct ggml_tensor * build_rope(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * pos, int n_dims, float theta = 10000.0f);

    struct ggml_tensor * build_self_attention(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * mask, struct ggml_tensor * pos,
        const SelfAttentionWeights & w, int num_heads, bool is_causal);

    struct ggml_tensor * build_mlp(
        struct ggml_context * ctx, struct ggml_tensor * x,
        const MLPWeights & w);

    // Returns {normed_x, gate}
    std::pair<struct ggml_tensor *, struct ggml_tensor *> build_adaln(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * cond_embed,
        const LowRankAdaLNWeights & w, float eps);

    struct ggml_tensor * build_joint_attention(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * attn_mask,
        struct ggml_tensor * pos, int start_pos,
        const EchoKVPair & kv_text, const EchoKVPair & kv_speaker,
        const JointAttentionWeights & w, int num_heads,
        int speaker_patch_size,
        const EchoKVPair * kv_latent = nullptr);

    struct ggml_tensor * build_encoder_block(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * mask, struct ggml_tensor * pos,
        const EncoderBlockWeights & w, int num_heads,
        bool is_causal, float eps);

    struct ggml_tensor * build_decoder_block(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * cond_embed,
        struct ggml_tensor * attn_mask,
        struct ggml_tensor * pos, int start_pos,
        const EchoKVPair & kv_text, const EchoKVPair & kv_speaker,
        const DecoderBlockWeights & w,
        const EchoKVPair * kv_latent = nullptr,
        int layer_index = -1,
        std::vector<std::pair<std::string, struct ggml_tensor *>> * debug_tensors = nullptr);

    // Weight loading
    struct ggml_tensor * get_tensor(const std::string & name);
    void load_hparams();
    void load_pca_state();
    void load_weights();
    void load_cond_cpu_weights();

    // Graph execution helpers
    struct ggml_context * create_compute_ctx(size_t n_tensors);
    void compute_graph(struct ggml_context * ctx, struct ggml_tensor * output);

    // Build a KV cache by running an encoder + per-layer KV projections
    EchoKVCache build_and_compute_kv_cache(
        const std::string & encoder_type,  // "text", "speaker", or "latent"
        const void * input_data,
        const void * mask_data,
        int seq_len,
        int batch_size
    );
};
