// Echo-TTS C++ — EchoModel Implementation
//
// GGUF weight loading, GGML graph building for the full EchoDiT architecture.
// This is the core file of the inference engine.

#include "echo_model.h"
#include "ggml-alloc.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cstdlib>

static void print_tensor_stats(const char * name, struct ggml_tensor * t) {
    size_t n = ggml_nelements(t);
    if (n == 0) {
        printf("[debug] %-18s empty\n", name);
        return;
    }

    std::vector<float> data(n);
    ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));

    double sum = 0.0;
    double sum_sq = 0.0;
    float mn = data[0];
    float mx = data[0];
    for (float v : data) {
        sum += v;
        sum_sq += (double)v * (double)v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }

    double mean = sum / (double)n;
    double var = std::max(0.0, sum_sq / (double)n - mean * mean);
    printf("[debug] %-18s shape=[%lld,%lld,%lld,%lld] mean=% .6f std=% .6f range=[% .6f,% .6f]\n",
           name,
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3],
           mean, std::sqrt(var), mn, mx);
}

static std::vector<float> tensor_to_float_vector(struct ggml_tensor * t) {
    size_t n = ggml_nelements(t);
    std::vector<float> out(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
    } else if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        const ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            int64_t blck_size = traits->blck_size;
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> raw(nbytes);
            ggml_backend_tensor_get(t, raw.data(), 0, nbytes);

            const uint8_t * src = raw.data();
            float * dst = out.data();
            int64_t n_blocks = (int64_t)n / blck_size;
            for (int64_t i = 0; i < n_blocks; i++) {
                traits->to_float(src, dst, blck_size);
                src += traits->type_size;
                dst += blck_size;
            }
        } else {
            fprintf(stderr, "[echo_model] ERROR: unsupported CPU weight cache type %d\n", (int)t->type);
        }
    } else {
        fprintf(stderr, "[echo_model] ERROR: unsupported CPU weight cache type %d\n", (int)t->type);
    }

    return out;
}

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// ────────────────────────────────────────────────────────────────────
// KV Cache lifecycle
// ────────────────────────────────────────────────────────────────────

void EchoKVCache::free() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        void * ctx_buf = ggml_get_mem_buffer(ctx);
        ggml_free(ctx);
        ctx = nullptr;
        delete[] static_cast<uint8_t *>(ctx_buf);
    }
    layers.clear();
    seq_len = 0;
    batch_size = 0;
}

EchoKVCache::EchoKVCache(EchoKVCache && o) noexcept
    : layers(std::move(o.layers))
    , ctx(o.ctx)
    , buffer(o.buffer)
    , seq_len(o.seq_len)
    , batch_size(o.batch_size)
{
    o.ctx = nullptr;
    o.buffer = nullptr;
    o.seq_len = 0;
    o.batch_size = 0;
}

EchoKVCache & EchoKVCache::operator=(EchoKVCache && o) noexcept {
    if (this != &o) {
        free();
        layers     = std::move(o.layers);
        ctx        = o.ctx;
        buffer     = o.buffer;
        seq_len    = o.seq_len;
        batch_size = o.batch_size;
        o.ctx = nullptr;
        o.buffer = nullptr;
        o.seq_len = 0;
        o.batch_size = 0;
    }
    return *this;
}

// ────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ────────────────────────────────────────────────────────────────────

EchoModel::EchoModel() {
    ggml_backend_load_all();
    backend_ = ggml_backend_init_best();
    if (!backend_) {
        fprintf(stderr, "[echo_model] WARNING: No GPU backend found, falling back to CPU\n");
        backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    cpu_backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    printf("[echo_model] Using backend: %s\n", ggml_backend_name(backend_));

    ggml_backend_t backends[] = { backend_, cpu_backend_ };
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, 8192, false, true);
}

EchoModel::~EchoModel() {
    if (sched_)       { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (weight_buf_)  { ggml_backend_buffer_free(weight_buf_); weight_buf_ = nullptr; }
    if (weight_ctx_)  { ggml_free(weight_ctx_); weight_ctx_ = nullptr; }
    if (gguf_ctx_)    { gguf_free(gguf_ctx_); gguf_ctx_ = nullptr; }
    if (backend_)     { ggml_backend_free(backend_); backend_ = nullptr; }
    if (cpu_backend_) { ggml_backend_free(cpu_backend_); cpu_backend_ = nullptr; }
}

// ────────────────────────────────────────────────────────────────────
// GGUF Loading
// ────────────────────────────────────────────────────────────────────

struct ggml_tensor * EchoModel::get_tensor(const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(weight_ctx_, name.c_str());
    if (!t) {
        fprintf(stderr, "[echo_model] WARNING: tensor '%s' not found\n", name.c_str());
    }
    return t;
}

void EchoModel::load_hparams() {
    auto read_u32 = [this](const char * key, int32_t & out) {
        int64_t id = gguf_find_key(gguf_ctx_, key);
        if (id >= 0) out = static_cast<int32_t>(gguf_get_val_u32(gguf_ctx_, id));
    };
    auto read_f32 = [this](const char * key, float & out) {
        int64_t id = gguf_find_key(gguf_ctx_, key);
        if (id >= 0) out = gguf_get_val_f32(gguf_ctx_, id);
    };
    auto read_bool = [this](const char * key, bool & out) {
        int64_t id = gguf_find_key(gguf_ctx_, key);
        if (id >= 0) out = gguf_get_val_bool(gguf_ctx_, id);
    };

    read_u32("echo.latent_size",             hparams_.latent_size);
    read_u32("echo.model_size",              hparams_.model_size);
    read_u32("echo.num_layers",              hparams_.num_layers);
    read_u32("echo.num_heads",               hparams_.num_heads);
    read_u32("echo.intermediate_size",        hparams_.intermediate_size);
    read_f32("echo.norm_eps",                hparams_.norm_eps);
    read_u32("echo.text_vocab_size",         hparams_.text_vocab_size);
    read_u32("echo.text_model_size",         hparams_.text_model_size);
    read_u32("echo.text_num_layers",         hparams_.text_num_layers);
    read_u32("echo.text_num_heads",          hparams_.text_num_heads);
    read_u32("echo.text_intermediate_size",  hparams_.text_intermediate_size);
    read_u32("echo.speaker_patch_size",      hparams_.speaker_patch_size);
    read_u32("echo.speaker_model_size",      hparams_.speaker_model_size);
    read_u32("echo.speaker_num_layers",      hparams_.speaker_num_layers);
    read_u32("echo.speaker_num_heads",       hparams_.speaker_num_heads);
    read_u32("echo.speaker_intermediate_size", hparams_.speaker_intermediate_size);
    read_u32("echo.timestep_embed_size",     hparams_.timestep_embed_size);
    read_u32("echo.adaln_rank",              hparams_.adaln_rank);
    read_bool("echo.include_blockwise",      hparams_.include_blockwise);

    printf("[echo_model] Loaded hparams: model_size=%d, num_layers=%d, num_heads=%d\n",
           hparams_.model_size, hparams_.num_layers, hparams_.num_heads);
    printf("[echo_model]   text: vocab=%d, layers=%d, heads=%d\n",
           hparams_.text_vocab_size, hparams_.text_num_layers, hparams_.text_num_heads);
    printf("[echo_model]   speaker: patch=%d, layers=%d, heads=%d\n",
           hparams_.speaker_patch_size, hparams_.speaker_num_layers, hparams_.speaker_num_heads);
}

void EchoModel::load_pca_state() {
    // Read PCA tensors (stored as F32 in the GGUF)
    struct ggml_tensor * comp_t = get_tensor("pca.components");
    struct ggml_tensor * mean_t = get_tensor("pca.mean");

    if (comp_t) {
        size_t n = ggml_nelements(comp_t);
        pca_state_.components.resize(n);
        ggml_backend_tensor_get(comp_t, pca_state_.components.data(), 0, n * sizeof(float));
    }
    if (mean_t) {
        size_t n = ggml_nelements(mean_t);
        pca_state_.mean.resize(n);
        ggml_backend_tensor_get(mean_t, pca_state_.mean.data(), 0, n * sizeof(float));
    }

    // Read latent_scale from metadata
    int64_t id = gguf_find_key(gguf_ctx_, "pca.latent_scale");
    if (id >= 0) {
        pca_state_.latent_scale = gguf_get_val_f32(gguf_ctx_, id);
    }

    printf("[echo_model] PCA state: components=%zu, mean=%zu, scale=%.4f\n",
           pca_state_.components.size(), pca_state_.mean.size(), pca_state_.latent_scale);
}

void EchoModel::load_weights() {
    auto & w = weights_;
    auto & hp = hparams_;

    // ── Text encoder ──
    w.text_embedding = get_tensor("text_encoder.text_embedding.weight");
    w.text_blocks.resize(hp.text_num_layers);
    for (int i = 0; i < hp.text_num_layers; i++) {
        std::string prefix = "text_encoder.blocks." + std::to_string(i);
        auto & b = w.text_blocks[i];
        b.attention.wq     = get_tensor(prefix + ".attention.wq.weight");
        b.attention.wk     = get_tensor(prefix + ".attention.wk.weight");
        b.attention.wv     = get_tensor(prefix + ".attention.wv.weight");
        b.attention.wo     = get_tensor(prefix + ".attention.wo.weight");
        b.attention.gate   = get_tensor(prefix + ".attention.gate.weight");
        b.attention.q_norm = get_tensor(prefix + ".attention.q_norm.weight");
        b.attention.k_norm = get_tensor(prefix + ".attention.k_norm.weight");
        b.mlp.w1           = get_tensor(prefix + ".mlp.w1.weight");
        b.mlp.w3           = get_tensor(prefix + ".mlp.w3.weight");
        b.mlp.w2           = get_tensor(prefix + ".mlp.w2.weight");
        b.attention_norm   = get_tensor(prefix + ".attention_norm.weight");
        b.mlp_norm         = get_tensor(prefix + ".mlp_norm.weight");
    }

    // ── Speaker encoder ──
    w.speaker_in_proj      = get_tensor("speaker_encoder.in_proj.weight");
    w.speaker_in_proj_bias = get_tensor("speaker_encoder.in_proj.bias");
    w.speaker_blocks.resize(hp.speaker_num_layers);
    for (int i = 0; i < hp.speaker_num_layers; i++) {
        std::string prefix = "speaker_encoder.blocks." + std::to_string(i);
        auto & b = w.speaker_blocks[i];
        b.attention.wq     = get_tensor(prefix + ".attention.wq.weight");
        b.attention.wk     = get_tensor(prefix + ".attention.wk.weight");
        b.attention.wv     = get_tensor(prefix + ".attention.wv.weight");
        b.attention.wo     = get_tensor(prefix + ".attention.wo.weight");
        b.attention.gate   = get_tensor(prefix + ".attention.gate.weight");
        b.attention.q_norm = get_tensor(prefix + ".attention.q_norm.weight");
        b.attention.k_norm = get_tensor(prefix + ".attention.k_norm.weight");
        b.mlp.w1           = get_tensor(prefix + ".mlp.w1.weight");
        b.mlp.w3           = get_tensor(prefix + ".mlp.w3.weight");
        b.mlp.w2           = get_tensor(prefix + ".mlp.w2.weight");
        b.attention_norm   = get_tensor(prefix + ".attention_norm.weight");
        b.mlp_norm         = get_tensor(prefix + ".mlp_norm.weight");
    }

    // ── Latent encoder (blockwise) ──
    if (hp.include_blockwise) {
        w.latent_in_proj      = get_tensor("latent_encoder.in_proj.weight");
        w.latent_in_proj_bias = get_tensor("latent_encoder.in_proj.bias");
        w.latent_blocks.resize(hp.speaker_num_layers);
        for (int i = 0; i < hp.speaker_num_layers; i++) {
            std::string prefix = "latent_encoder.blocks." + std::to_string(i);
            auto & b = w.latent_blocks[i];
            b.attention.wq     = get_tensor(prefix + ".attention.wq.weight");
            b.attention.wk     = get_tensor(prefix + ".attention.wk.weight");
            b.attention.wv     = get_tensor(prefix + ".attention.wv.weight");
            b.attention.wo     = get_tensor(prefix + ".attention.wo.weight");
            b.attention.gate   = get_tensor(prefix + ".attention.gate.weight");
            b.attention.q_norm = get_tensor(prefix + ".attention.q_norm.weight");
            b.attention.k_norm = get_tensor(prefix + ".attention.k_norm.weight");
            b.mlp.w1           = get_tensor(prefix + ".mlp.w1.weight");
            b.mlp.w3           = get_tensor(prefix + ".mlp.w3.weight");
            b.mlp.w2           = get_tensor(prefix + ".mlp.w2.weight");
            b.attention_norm   = get_tensor(prefix + ".attention_norm.weight");
            b.mlp_norm         = get_tensor(prefix + ".mlp_norm.weight");
        }
    }

    // ── Post-encoder norms ──
    w.text_norm    = get_tensor("text_norm.weight");
    w.speaker_norm = get_tensor("speaker_norm.weight");
    if (hp.include_blockwise) {
        w.latent_norm = get_tensor("latent_norm.weight");
    }

    // ── Conditioning module ──
    w.cond_linear0 = get_tensor("cond_module.0.weight");
    w.cond_linear1 = get_tensor("cond_module.2.weight");
    w.cond_linear2 = get_tensor("cond_module.4.weight");

    // ── Input projection ──
    w.in_proj      = get_tensor("in_proj.weight");
    w.in_proj_bias = get_tensor("in_proj.bias");

    // ── Decoder blocks ──
    w.decoder_blocks.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        std::string prefix = "blocks." + std::to_string(i);
        auto & b = w.decoder_blocks[i];

        // Joint attention
        b.attention.wq         = get_tensor(prefix + ".attention.wq.weight");
        b.attention.wk         = get_tensor(prefix + ".attention.wk.weight");
        b.attention.wv         = get_tensor(prefix + ".attention.wv.weight");
        b.attention.wk_text    = get_tensor(prefix + ".attention.wk_text.weight");
        b.attention.wv_text    = get_tensor(prefix + ".attention.wv_text.weight");
        b.attention.wk_speaker = get_tensor(prefix + ".attention.wk_speaker.weight");
        b.attention.wv_speaker = get_tensor(prefix + ".attention.wv_speaker.weight");
        if (hp.include_blockwise) {
            b.attention.wk_latent = get_tensor(prefix + ".attention.wk_latent.weight");
            b.attention.wv_latent = get_tensor(prefix + ".attention.wv_latent.weight");
        }
        b.attention.q_norm     = get_tensor(prefix + ".attention.q_norm.weight");
        b.attention.k_norm     = get_tensor(prefix + ".attention.k_norm.weight");
        b.attention.gate       = get_tensor(prefix + ".attention.gate.weight");
        b.attention.wo         = get_tensor(prefix + ".attention.wo.weight");

        // MLP
        b.mlp.w1 = get_tensor(prefix + ".mlp.w1.weight");
        b.mlp.w3 = get_tensor(prefix + ".mlp.w3.weight");
        b.mlp.w2 = get_tensor(prefix + ".mlp.w2.weight");

        // AdaLN (attention)
        auto load_adaln = [&](const std::string & aprefix, LowRankAdaLNWeights & a) {
            a.shift_down    = get_tensor(aprefix + ".shift_down.weight");
            a.scale_down    = get_tensor(aprefix + ".scale_down.weight");
            a.gate_down     = get_tensor(aprefix + ".gate_down.weight");
            a.shift_up      = get_tensor(aprefix + ".shift_up.weight");
            a.scale_up      = get_tensor(aprefix + ".scale_up.weight");
            a.gate_up       = get_tensor(aprefix + ".gate_up.weight");
            a.shift_up_bias = get_tensor(aprefix + ".shift_up.bias");
            a.scale_up_bias = get_tensor(aprefix + ".scale_up.bias");
            a.gate_up_bias  = get_tensor(aprefix + ".gate_up.bias");
        };
        load_adaln(prefix + ".attention_adaln", b.attention_adaln);
        load_adaln(prefix + ".mlp_adaln",       b.mlp_adaln);
    }

    // ── Output ──
    w.out_norm      = get_tensor("out_norm.weight");
    w.out_proj      = get_tensor("out_proj.weight");
    w.out_proj_bias = get_tensor("out_proj.bias");

    printf("[echo_model] All weight pointers loaded\n");
}

void EchoModel::load_cond_cpu_weights() {
    cond_w0_cpu_ = tensor_to_float_vector(weights_.cond_linear0);
    cond_w1_cpu_ = tensor_to_float_vector(weights_.cond_linear1);
    cond_w2_cpu_ = tensor_to_float_vector(weights_.cond_linear2);
}

bool EchoModel::load(const std::string & path) {
    printf("[echo_model] Loading model from: %s\n", path.c_str());

    // Open GGUF (no_alloc: tensor metadata only, no data yet)
    struct gguf_init_params gguf_params = { /*.no_alloc=*/ true, /*.ctx=*/ &weight_ctx_ };
    gguf_ctx_ = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx_) {
        fprintf(stderr, "[echo_model] ERROR: failed to open GGUF file: %s\n", path.c_str());
        return false;
    }

    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    printf("[echo_model] GGUF: %lld tensors, data_offset=%zu\n",
           (long long)n_tensors, gguf_get_data_offset(gguf_ctx_));

    // Read hyperparameters and PCA metadata
    load_hparams();

    // Allocate all weight tensors on the GPU backend
    weight_buf_ = ggml_backend_alloc_ctx_tensors(weight_ctx_, backend_);
    if (!weight_buf_) {
        fprintf(stderr, "[echo_model] ERROR: failed to allocate weight tensors on backend\n");
        return false;
    }
    printf("[echo_model] Weight buffer: %.1f MB\n",
           ggml_backend_buffer_get_size(weight_buf_) / (1024.0 * 1024.0));

    // Load tensor data from the GGUF file
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[echo_model] ERROR: cannot open file for reading: %s\n", path.c_str());
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx_);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx_, i);
        struct ggml_tensor * tensor = ggml_get_tensor(weight_ctx_, name);
        if (!tensor) continue;

        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx_, i);
        size_t tensor_size   = ggml_nbytes(tensor);

        if (read_buf.size() < tensor_size) {
            read_buf.resize(tensor_size);
        }

#ifdef _WIN32
        _fseeki64(f, static_cast<int64_t>(data_offset + tensor_offset), SEEK_SET);
#else
        fseek(f, data_offset + tensor_offset, SEEK_SET);
#endif
        size_t nread = fread(read_buf.data(), 1, tensor_size, f);
        if (nread != tensor_size) {
            fprintf(stderr, "[echo_model] WARNING: short read for tensor '%s': %zu/%zu\n",
                    name, nread, tensor_size);
        }
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, tensor_size);
    }
    fclose(f);
    printf("[echo_model] Loaded %lld tensors from GGUF\n", (long long)n_tensors);

    // Load PCA state (copies from GPU tensors to CPU vectors)
    load_pca_state();

    // Wire all weight pointers
    load_weights();
    load_cond_cpu_weights();

    return true;
}

// ────────────────────────────────────────────────────────────────────
// Graph execution helpers
// ────────────────────────────────────────────────────────────────────

struct ggml_context * EchoModel::create_compute_ctx(size_t n_tensors) {
    size_t buf_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false);
    auto * buf = new uint8_t[buf_size];
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    return ctx;
}

void EchoModel::compute_graph(struct ggml_context * ctx, struct ggml_tensor * output) {
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, output);

    ggml_backend_sched_reset(sched_);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        fprintf(stderr, "[echo_model] ERROR: failed to allocate graph\n");
        return;
    }

    ggml_backend_sched_graph_compute(sched_, gf);
}

// ────────────────────────────────────────────────────────────────────
// Graph-building helpers: primitives
// ────────────────────────────────────────────────────────────────────

struct ggml_tensor * EchoModel::build_rms_norm(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * weight, float eps
) {
    // rms_norm(x) * weight
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, weight);
}

struct ggml_tensor * EchoModel::build_rope(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * pos, int n_dims, float theta
) {
    // x shape: [head_dim, seq, num_heads, batch] or 3D
    return ggml_rope_ext(
        ctx, x, pos,
        nullptr,                // freq_factors
        n_dims,
        GGML_ROPE_TYPE_NORMAL,
        0,                      // n_ctx_orig
        theta,                  // freq_base
        1.0f,                   // freq_scale
        0.0f, 1.0f, 0.0f, 0.0f // ext/attn/beta params
    );
}

struct ggml_tensor * EchoModel::build_self_attention(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * mask, struct ggml_tensor * pos,
    const SelfAttentionWeights & w, int num_heads, bool is_causal
) {
    // x: [model_size, seq, batch] in GGML layout
    int64_t model_size = x->ne[0];
    int64_t seq_len    = x->ne[1];
    int64_t batch      = x->ne[2] > 0 ? x->ne[2] : 1;
    int64_t head_dim   = model_size / num_heads;

    // Q, K, V projections: [model_size, seq, batch]
    struct ggml_tensor * q = ggml_mul_mat(ctx, w.wq, x);
    struct ggml_tensor * k = ggml_mul_mat(ctx, w.wk, x);
    struct ggml_tensor * v = ggml_mul_mat(ctx, w.wv, x);

    // Gate: [model_size, seq, batch]
    struct ggml_tensor * gate = ggml_mul_mat(ctx, w.gate, x);

    // Reshape to [head_dim, num_heads, seq, batch]
    // GGML: after mul_mat, shape is [model_size, seq, batch]
    q = ggml_reshape_4d(ctx, q, head_dim, num_heads, seq_len, batch);
    k = ggml_reshape_4d(ctx, k, head_dim, num_heads, seq_len, batch);
    v = ggml_reshape_4d(ctx, v, head_dim, num_heads, seq_len, batch);

    // QK-norm: weight shape is [num_heads, head_dim] in PyTorch = [head_dim, num_heads] in GGML
    // rms_norm over last dim (head_dim in this layout), then mul by 2D weight
    q = ggml_rms_norm(ctx, q, hparams_.norm_eps);
    struct ggml_tensor * q_norm_reshaped = ggml_reshape_4d(ctx, w.q_norm, head_dim, num_heads, 1, 1);
    q = ggml_mul(ctx, q, q_norm_reshaped);
    
    k = ggml_rms_norm(ctx, k, hparams_.norm_eps);
    struct ggml_tensor * k_norm_reshaped = ggml_reshape_4d(ctx, w.k_norm, head_dim, num_heads, 1, 1);
    k = ggml_mul(ctx, k, k_norm_reshaped);

    // RoPE
    q = build_rope(ctx, q, pos, (int)head_dim);
    k = build_rope(ctx, k, pos, (int)head_dim);

    // Permute to [head_dim, seq, num_heads, batch] for Flash Attention
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Flash attention
    // Q: [head_dim, seq, num_heads, batch]
    // K: [head_dim, seq, num_heads, batch]
    // V: [head_dim, seq, num_heads, batch]
    float scale = 1.0f / sqrtf((float)head_dim);

    // Use the pre-built mask tensor directly (created at encoder level)
    // mask is either: causal F16 [seq,seq], bidirectional F16 [seq,seq,1,batch], or nullptr
    struct ggml_tensor * attn_mask = mask;

    struct ggml_tensor * attn_out = ggml_flash_attn_ext(
        ctx, q, k, v, attn_mask, scale, 0.0f, 0.0f
    );
    ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

    // Flash attention output: [head_dim, num_heads, seq, batch]
    // Reshape directly to [model_size, seq, batch] (merge head_dim + num_heads)
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_3d(ctx, attn_out, model_size, seq_len, batch);

    // Gated attention: output * sigmoid(gate)
    gate = ggml_sigmoid(ctx, gate);
    attn_out = ggml_mul(ctx, attn_out, gate);

    // Output projection
    attn_out = ggml_mul_mat(ctx, w.wo, attn_out);

    return attn_out;
}

struct ggml_tensor * EchoModel::build_mlp(
    struct ggml_context * ctx, struct ggml_tensor * x,
    const MLPWeights & w
) {
    // SwiGLU: w2(silu(w1(x)) * w3(x))
    struct ggml_tensor * h1 = ggml_mul_mat(ctx, w.w1, x);
    struct ggml_tensor * h3 = ggml_mul_mat(ctx, w.w3, x);
    h1 = ggml_silu(ctx, h1);
    struct ggml_tensor * h = ggml_mul(ctx, h1, h3);
    return ggml_mul_mat(ctx, w.w2, h);
}

std::pair<struct ggml_tensor *, struct ggml_tensor *> EchoModel::build_adaln(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * cond_embed,
    const LowRankAdaLNWeights & w, float eps
) {
    // cond_embed: [model_size*3, 1, batch] (already unsqueezed)
    // Split into shift, scale, gate — each [model_size, 1, batch]
    int64_t model_size = x->ne[0];

    // ggml_view_3d to slice the 3 chunks along ne[0]
    // cond_embed has shape [model_size*3, 1, batch]
    int64_t nb1 = cond_embed->nb[1];
    int64_t nb2 = cond_embed->nb[2];

    struct ggml_tensor * shift_in = ggml_view_3d(ctx, cond_embed,
        model_size, 1, cond_embed->ne[2],
        nb1, nb2, 0);
    struct ggml_tensor * scale_in = ggml_view_3d(ctx, cond_embed,
        model_size, 1, cond_embed->ne[2],
        nb1, nb2, model_size * ggml_element_size(cond_embed));
    struct ggml_tensor * gate_in  = ggml_view_3d(ctx, cond_embed,
        model_size, 1, cond_embed->ne[2],
        nb1, nb2, 2 * model_size * ggml_element_size(cond_embed));

    // LoRA with residual: shift = shift_up(shift_down(silu(shift_in))) + shift_in
    auto lora_residual = [&](struct ggml_tensor * input,
                             struct ggml_tensor * down_w,
                             struct ggml_tensor * up_w,
                             struct ggml_tensor * up_bias) -> struct ggml_tensor * {
        struct ggml_tensor * h = ggml_silu(ctx, input);
        h = ggml_mul_mat(ctx, down_w, h);
        h = ggml_mul_mat(ctx, up_w, h);
        h = ggml_add(ctx, h, up_bias);
        h = ggml_add(ctx, h, input);   // residual
        return h;
    };

    struct ggml_tensor * shift = lora_residual(shift_in, w.shift_down, w.shift_up, w.shift_up_bias);
    struct ggml_tensor * scale = lora_residual(scale_in, w.scale_down, w.scale_up, w.scale_up_bias);
    struct ggml_tensor * gate  = lora_residual(gate_in,  w.gate_down,  w.gate_up,  w.gate_up_bias);

    // x_norm = rms_norm(x) * (scale + 1) + shift
    struct ggml_tensor * x_norm = ggml_rms_norm(ctx, x, eps);
    struct ggml_tensor * scaled = ggml_mul(ctx, x_norm, scale);
    x_norm = ggml_add(ctx, scaled, x_norm);
    x_norm = ggml_add(ctx, x_norm, shift);

    // gate = tanh(gate)
    gate = ggml_tanh(ctx, gate);

    return { x_norm, gate };
}

// ────────────────────────────────────────────────────────────────────
// Graph-building helpers: composite blocks
// ────────────────────────────────────────────────────────────────────

struct ggml_tensor * EchoModel::build_joint_attention(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * attn_mask,
    struct ggml_tensor * pos, int start_pos,
    const EchoKVPair & kv_text, const EchoKVPair & kv_speaker,
    const JointAttentionWeights & w, int num_heads,
    int speaker_patch_size,
    const EchoKVPair * kv_latent
) {
    // x: [model_size, seq, batch]
    int64_t model_size = x->ne[0];
    int64_t seq_len    = x->ne[1];
    int64_t batch      = x->ne[2] > 0 ? x->ne[2] : 1;
    int64_t head_dim   = model_size / num_heads;
    int64_t half_heads = num_heads / 2;

    // Self Q, K, V projections
    struct ggml_tensor * q      = ggml_mul_mat(ctx, w.wq, x);
    struct ggml_tensor * k_self = ggml_mul_mat(ctx, w.wk, x);
    struct ggml_tensor * v_self = ggml_mul_mat(ctx, w.wv, x);

    // Gate: [model_size, seq, batch]
    struct ggml_tensor * gate = ggml_mul_mat(ctx, w.gate, x);

    // Reshape to [head_dim, num_heads, seq, batch]
    q      = ggml_reshape_4d(ctx, q,      head_dim, num_heads, seq_len, batch);
    k_self = ggml_reshape_4d(ctx, k_self, head_dim, num_heads, seq_len, batch);
    v_self = ggml_reshape_4d(ctx, v_self, head_dim, num_heads, seq_len, batch);

    // QK-norm (uses decoder's q_norm/k_norm which have shape [num_heads, head_dim])
    q      = ggml_rms_norm(ctx, q, hparams_.norm_eps);
    struct ggml_tensor * q_norm_reshaped = ggml_reshape_4d(ctx, w.q_norm, head_dim, num_heads, 1, 1);
    q      = ggml_mul(ctx, q, q_norm_reshaped);
    
    k_self = ggml_rms_norm(ctx, k_self, hparams_.norm_eps);
    struct ggml_tensor * k_norm_reshaped = ggml_reshape_4d(ctx, w.k_norm, head_dim, num_heads, 1, 1);
    k_self = ggml_mul(ctx, k_self, k_norm_reshaped);

    // ── Half-RoPE ──
    // Split heads in half: apply RoPE to first half, leave second half unchanged.
    // In GGML 4D [head_dim, num_heads, seq, batch], split along dim 1 (num_heads).
    struct ggml_tensor * q1 = ggml_view_4d(ctx, q,
        head_dim, half_heads, seq_len, batch,
        q->nb[1], q->nb[2], q->nb[3], 0);
    struct ggml_tensor * q2 = ggml_view_4d(ctx, q,
        head_dim, half_heads, seq_len, batch,
        q->nb[1], q->nb[2], q->nb[3],
        half_heads * q->nb[1]);
    q1 = build_rope(ctx, q1, pos, (int)head_dim);
    q = ggml_concat(ctx, q1, q2, 1);  // concat along num_heads dim

    struct ggml_tensor * k1 = ggml_view_4d(ctx, k_self,
        head_dim, half_heads, seq_len, batch,
        k_self->nb[1], k_self->nb[2], k_self->nb[3], 0);
    struct ggml_tensor * k2 = ggml_view_4d(ctx, k_self,
        head_dim, half_heads, seq_len, batch,
        k_self->nb[1], k_self->nb[2], k_self->nb[3],
        half_heads * k_self->nb[1]);
    k1 = build_rope(ctx, k1, pos, (int)head_dim);
    k_self = ggml_concat(ctx, k1, k2, 1);

    // Permute to [head_dim, seq, num_heads, batch] for Flash Attention
    q      = ggml_permute(ctx, q,      0, 2, 1, 3);
    k_self = ggml_permute(ctx, k_self, 0, 2, 1, 3);
    v_self = ggml_permute(ctx, v_self, 0, 2, 1, 3);

    // ── Concatenate all K and V sources ──
    // Order: [self, latent (if any), text, speaker]
    // This matches the Python: torch.cat([xk_self, xk_latent, xk_text, xk_speaker], dim=1)
    // In GGML: concat along dim 1 (seq dimension in [head_dim, seq, heads, batch])

    struct ggml_tensor * k_full = k_self;
    struct ggml_tensor * v_full = v_self;

    // Latent KV (blockwise only)
    bool has_latent = (kv_latent != nullptr && kv_latent->k != nullptr && kv_latent->k->ne[1] > 0);
    if (has_latent) {
        k_full = ggml_concat(ctx, k_full, kv_latent->k, 1);
        v_full = ggml_concat(ctx, v_full, kv_latent->v, 1);
    }

    // Text KV
    k_full = ggml_concat(ctx, k_full, kv_text.k, 1);
    v_full = ggml_concat(ctx, v_full, kv_text.v, 1);

    // Speaker KV
    k_full = ggml_concat(ctx, k_full, kv_speaker.k, 1);
    v_full = ggml_concat(ctx, v_full, kv_speaker.v, 1);

    // ── Build combined attention mask ──
    // Mask shape: [total_kv_len, seq_len, 1, batch]
    // self_mask: all attend (0.0)
    // latent_mask: attend only where position < start_pos
    // text_mask: from input (0.0=attend, -inf=mask)
    // speaker_mask: from input
    // The mask is built externally and passed in as attn_mask.

    // Flash attention
    float scale = 1.0f / sqrtf((float)head_dim);
    struct ggml_tensor * attn_out = ggml_flash_attn_ext(
        ctx, q, k_full, v_full, attn_mask, scale, 0.0f, 0.0f
    );
    ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

    // Output: [head_dim, num_heads, seq, batch] → [model_size, seq, batch]
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_3d(ctx, attn_out, model_size, seq_len, batch);

    // Gated attention
    gate = ggml_sigmoid(ctx, gate);
    attn_out = ggml_mul(ctx, attn_out, gate);

    // Output projection
    attn_out = ggml_mul_mat(ctx, w.wo, attn_out);

    return attn_out;
}

struct ggml_tensor * EchoModel::build_encoder_block(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * mask, struct ggml_tensor * pos,
    const EncoderBlockWeights & w, int num_heads,
    bool is_causal, float eps
) {
    // Pre-norm residual: x = x + attn(norm(x))
    struct ggml_tensor * x_norm = build_rms_norm(ctx, x, w.attention_norm, eps);
    struct ggml_tensor * attn_out = build_self_attention(
        ctx, x_norm, mask, pos, w.attention, num_heads, is_causal
    );
    x = ggml_add(ctx, x, attn_out);

    // Pre-norm residual: x = x + mlp(norm(x))
    x_norm = build_rms_norm(ctx, x, w.mlp_norm, eps);
    struct ggml_tensor * mlp_out = build_mlp(ctx, x_norm, w.mlp);
    x = ggml_add(ctx, x, mlp_out);

    return x;
}

struct ggml_tensor * EchoModel::build_decoder_block(
    struct ggml_context * ctx, struct ggml_tensor * x,
    struct ggml_tensor * cond_embed,
    struct ggml_tensor * attn_mask,
    struct ggml_tensor * pos, int start_pos,
    const EchoKVPair & kv_text, const EchoKVPair & kv_speaker,
    const DecoderBlockWeights & w,
    const EchoKVPair * kv_latent,
    int layer_index,
    std::vector<std::pair<std::string, struct ggml_tensor *>> * debug_tensors
) {
    auto mark_debug = [&](const std::string & name, struct ggml_tensor * t) {
        if (debug_tensors && layer_index == 0) {
            ggml_set_output(t);
            debug_tensors->push_back({name, t});
        }
    };

    // ── Attention branch with AdaLN ──
    auto [x_norm_attn, attn_gate] = build_adaln(
        ctx, x, cond_embed, w.attention_adaln, hparams_.norm_eps
    );
    mark_debug("block0_adaln_attn_x", x_norm_attn);
    mark_debug("block0_adaln_attn_gate", attn_gate);

    struct ggml_tensor * attn_out = build_joint_attention(
        ctx, x_norm_attn, attn_mask, pos, start_pos,
        kv_text, kv_speaker, w.attention, hparams_.num_heads,
        hparams_.speaker_patch_size, kv_latent
    );
    mark_debug("block0_attn_out", attn_out);

    // x = x + gate * attn_out
    attn_out = ggml_mul(ctx, attn_out, attn_gate);
    mark_debug("block0_attn_gated", attn_out);
    x = ggml_add(ctx, x, attn_out);
    mark_debug("block0_after_attn", x);

    // ── MLP branch with AdaLN ──
    auto [x_norm_mlp, mlp_gate] = build_adaln(
        ctx, x, cond_embed, w.mlp_adaln, hparams_.norm_eps
    );
    mark_debug("block0_adaln_mlp_x", x_norm_mlp);
    mark_debug("block0_adaln_mlp_gate", mlp_gate);

    struct ggml_tensor * mlp_out = build_mlp(ctx, x_norm_mlp, w.mlp);
    mark_debug("block0_mlp_out", mlp_out);

    // x = x + gate * mlp_out
    mlp_out = ggml_mul(ctx, mlp_out, mlp_gate);
    mark_debug("block0_mlp_gated", mlp_out);
    x = ggml_add(ctx, x, mlp_out);

    return x;
}

// ────────────────────────────────────────────────────────────────────
// KV Cache computation
// ────────────────────────────────────────────────────────────────────

EchoKVCache EchoModel::build_and_compute_kv_cache(
    const std::string & encoder_type,
    const void * input_data,
    const void * mask_data,
    int seq_len,
    int batch_size
) {
    auto & hp = hparams_;
    auto & w  = weights_;
    bool is_text    = (encoder_type == "text");
    bool is_speaker = (encoder_type == "speaker");
    bool is_latent  = (encoder_type == "latent");

    int enc_model_size = is_text ? hp.text_model_size : hp.speaker_model_size;
    int enc_num_heads  = is_text ? hp.text_num_heads  : hp.speaker_num_heads;
    int enc_num_layers = is_text ? hp.text_num_layers  : hp.speaker_num_layers;
    int enc_head_dim   = enc_model_size / enc_num_heads;
    int dec_head_dim   = hp.head_dim();
    int dec_num_heads  = hp.num_heads;

    // For speaker/latent: input is (batch, seq_len, 80)
    // After patching: seq_len_patched = seq_len / patch_size
    int patched_seq_len = seq_len;
    if (!is_text) {
        patched_seq_len = seq_len / hp.speaker_patch_size;
    }

    // ── Step 1: Run encoder ──
    // Estimate tensor count: encoder blocks + input + norms + output
    size_t n_tensors = 2048;  // generous estimate
    struct ggml_context * ctx = create_compute_ctx(n_tensors);

    struct ggml_tensor * encoder_out;

    if (is_text) {
        // Text encoder: embedding lookup + 14 blocks + norm
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, seq_len, batch_size);
        ggml_set_input(ids);
        ggml_set_name(ids, "text_ids");

        // Embedding
        struct ggml_tensor * x = ggml_get_rows(ctx, w.text_embedding, ids);

        // Positions
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        ggml_set_input(pos);
        ggml_set_name(pos, "enc_pos");

        // Create a single bidirectional attention mask for all encoder layers
        // mask shape: [seq, seq, 1, batch] F16 (0=attend, -inf=mask)
        struct ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, seq_len, seq_len, 1, batch_size);
        ggml_set_input(attn_mask);
        ggml_set_name(attn_mask, "enc_attn_mask");

        // Encoder blocks (bidirectional with mask)
        for (int i = 0; i < enc_num_layers; i++) {
            x = build_encoder_block(ctx, x, attn_mask, pos,
                                    w.text_blocks[i], enc_num_heads, false, hp.norm_eps);
        }

        // Post-encoder norm
        encoder_out = build_rms_norm(ctx, x, w.text_norm, hp.norm_eps);
    } else {
        // Speaker/Latent encoder: patch + in_proj + /6 + 14 blocks + norm
        struct ggml_tensor * latent = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            hp.latent_size, seq_len, batch_size);
        ggml_set_input(latent);
        ggml_set_name(latent, is_speaker ? "speaker_latent" : "latent_prefix");

        // Patch: reshape (B, S, 80) → (B, S/P, 80*P) then linear
        // In GGML: [80, S, B] → [80*P, S/P, B]
        struct ggml_tensor * x = ggml_reshape_3d(ctx, latent,
            hp.latent_size * hp.speaker_patch_size,
            patched_seq_len,
            batch_size);

        // in_proj (linear with bias)
        struct ggml_tensor * in_proj_w = is_speaker ? w.speaker_in_proj : w.latent_in_proj;
        struct ggml_tensor * in_proj_b = is_speaker ? w.speaker_in_proj_bias : w.latent_in_proj_bias;
        x = ggml_mul_mat(ctx, in_proj_w, x);
        x = ggml_add(ctx, x, in_proj_b);

        // Scale by 1/6
        x = ggml_scale(ctx, x, 1.0f / 6.0f);

        // Positions
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, patched_seq_len);
        ggml_set_input(pos);
        ggml_set_name(pos, "enc_pos");

        // Create a single causal mask for all encoder layers
        struct ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, patched_seq_len, patched_seq_len);
        ggml_set_input(causal_mask);
        ggml_set_name(causal_mask, "enc_causal_mask");

        // Encoder blocks (causal)
        auto & blocks = is_speaker ? w.speaker_blocks : w.latent_blocks;
        for (int i = 0; i < enc_num_layers; i++) {
            x = build_encoder_block(ctx, x, causal_mask, pos,
                                    blocks[i], enc_num_heads, true, hp.norm_eps);
        }

        // Post-encoder norm
        struct ggml_tensor * norm_w = is_speaker ? w.speaker_norm : w.latent_norm;
        encoder_out = build_rms_norm(ctx, x, norm_w, hp.norm_eps);
    }

    // ── Step 2: Per-decoder-layer KV projections ──
    // For each decoder layer: project encoder output → K,V with k_norm
    // Output: num_layers × {K, V} each of shape [head_dim, patched_seq_len, dec_num_heads, batch]
    std::vector<struct ggml_tensor *> k_tensors(hp.num_layers);
    std::vector<struct ggml_tensor *> v_tensors(hp.num_layers);

    for (int i = 0; i < hp.num_layers; i++) {
        auto & dec_attn = w.decoder_blocks[i].attention;

        struct ggml_tensor * wk, * wv;
        if (is_text) {
            wk = dec_attn.wk_text;
            wv = dec_attn.wv_text;
        } else if (is_speaker) {
            wk = dec_attn.wk_speaker;
            wv = dec_attn.wv_speaker;
        } else {
            wk = dec_attn.wk_latent;
            wv = dec_attn.wv_latent;
        }

        // K projection + reshape + k_norm
        struct ggml_tensor * k = ggml_mul_mat(ctx, wk, encoder_out);
        k = ggml_reshape_4d(ctx, k, dec_head_dim, dec_num_heads, patched_seq_len, batch_size);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);  // [head_dim, seq, heads, batch]
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        // k is [head_dim, seq, heads, batch] after permute
        // k_norm is [head_dim, num_heads] — reshape to [head_dim, 1, num_heads, 1] for broadcasting
        struct ggml_tensor * k_norm_4d = ggml_reshape_4d(ctx, dec_attn.k_norm, dec_head_dim, 1, dec_num_heads, 1);
        k = ggml_mul(ctx, k, k_norm_4d);

        // For latent: apply half-RoPE with dilated positions
        if (is_latent) {
            int64_t half_heads = dec_num_heads / 2;
            // Create dilated positions: pos * speaker_patch_size
            struct ggml_tensor * lat_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, patched_seq_len);
            ggml_set_input(lat_pos);
            ggml_set_name(lat_pos, ("lat_pos_" + std::to_string(i)).c_str());

            struct ggml_tensor * k1_lat = ggml_view_4d(ctx, k,
                dec_head_dim, patched_seq_len, half_heads, batch_size,
                k->nb[1], k->nb[2], k->nb[3], 0);
            struct ggml_tensor * k2_lat = ggml_view_4d(ctx, k,
                dec_head_dim, patched_seq_len, half_heads, batch_size,
                k->nb[1], k->nb[2], k->nb[3],
                half_heads * k->nb[2]);
            k1_lat = build_rope(ctx, k1_lat, lat_pos, dec_head_dim);
            k = ggml_concat(ctx, k1_lat, k2_lat, 2);
        }

        k = ggml_cont(ctx, k);
        ggml_set_output(k);
        ggml_set_name(k, ("kv_k_" + std::to_string(i)).c_str());

        // V projection + reshape (no norm on V)
        struct ggml_tensor * v = ggml_mul_mat(ctx, wv, encoder_out);
        v = ggml_reshape_4d(ctx, v, dec_head_dim, dec_num_heads, patched_seq_len, batch_size);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        v = ggml_cont(ctx, v);
        ggml_set_output(v);
        ggml_set_name(v, ("kv_v_" + std::to_string(i)).c_str());

        k_tensors[i] = k;
        v_tensors[i] = v;
    }

    // ── Step 3: Build and compute the graph ──
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    for (int i = 0; i < hp.num_layers; i++) {
        ggml_build_forward_expand(gf, k_tensors[i]);
        ggml_build_forward_expand(gf, v_tensors[i]);
    }

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        fprintf(stderr, "[echo_model] ERROR: failed to allocate KV cache graph\n");
        ggml_free(ctx);
        return EchoKVCache();
    }

    // Set input data
    if (is_text) {
        struct ggml_tensor * ids_t = ggml_get_tensor(ctx, "text_ids");
        ggml_backend_tensor_set(ids_t, input_data, 0, seq_len * batch_size * sizeof(int32_t));
    } else {
        const char * latent_name = is_speaker ? "speaker_latent" : "latent_prefix";
        struct ggml_tensor * lat_t = ggml_get_tensor(ctx, latent_name);
        ggml_backend_tensor_set(lat_t, input_data, 0,
            (size_t)hp.latent_size * seq_len * batch_size * sizeof(float));
    }

    // Set encoder positions (sequential: 0, 1, 2, ...)
    {
        std::vector<int32_t> positions(patched_seq_len);
        for (int i = 0; i < patched_seq_len; i++) {
            positions[i] = i;
        }
        struct ggml_tensor * pos_t = ggml_get_tensor(ctx, "enc_pos");
        if (pos_t) {
            ggml_backend_tensor_set(pos_t, positions.data(), 0, patched_seq_len * sizeof(int32_t));
        }
    }

    // Set encoder attention mask
    if (is_text) {
        // Bidirectional mask from text_mask_data: [seq, seq, 1, batch] F16
        struct ggml_tensor * mask_t = ggml_get_tensor(ctx, "enc_attn_mask");
        if (mask_t && mask_data) {
            const float * text_mask_src = static_cast<const float *>(mask_data);
            std::vector<ggml_fp16_t> mask_buf((size_t)seq_len * seq_len * batch_size);
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-65000.0f);
            ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int b = 0; b < batch_size; b++) {
                for (int q = 0; q < seq_len; q++) {
                    for (int k = 0; k < seq_len; k++) {
                        float m = text_mask_src[b * seq_len + k];
                        mask_buf[b * seq_len * seq_len + q * seq_len + k] = (m > 0.5f) ? zero_val : neg_inf;
                    }
                }
            }
            ggml_backend_tensor_set(mask_t, mask_buf.data(), 0, mask_buf.size() * sizeof(ggml_fp16_t));
        }
    } else {
        // Causal mask: [seq, seq] F16
        struct ggml_tensor * mask_t = ggml_get_tensor(ctx, "enc_causal_mask");
        if (mask_t) {
            std::vector<ggml_fp16_t> mask_buf((size_t)patched_seq_len * patched_seq_len);
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-65000.0f);
            ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q = 0; q < patched_seq_len; q++) {
                for (int k = 0; k < patched_seq_len; k++) {
                    mask_buf[q * patched_seq_len + k] = (k > q) ? neg_inf : zero_val;
                }
            }
            ggml_backend_tensor_set(mask_t, mask_buf.data(), 0, mask_buf.size() * sizeof(ggml_fp16_t));
        }
    }

    // Set latent dilated position tensors (for latent encoder only)
    if (is_latent) {
        std::vector<int32_t> dilated_positions(patched_seq_len);
        for (int i = 0; i < patched_seq_len; i++) {
            dilated_positions[i] = i * hp.speaker_patch_size;
        }
        for (int i = 0; i < hp.num_layers; i++) {
            std::string name = "lat_pos_" + std::to_string(i);
            struct ggml_tensor * lat_pos_t = ggml_get_tensor(ctx, name.c_str());
            if (lat_pos_t) {
                ggml_backend_tensor_set(lat_pos_t, dilated_positions.data(), 0,
                    patched_seq_len * sizeof(int32_t));
            }
        }
    }

    ggml_backend_sched_graph_compute(sched_, gf);

    // ── Step 4: Copy results into a persistent KV cache ──
    EchoKVCache cache;
    cache.seq_len = patched_seq_len;
    cache.batch_size = batch_size;
    cache.layers.resize(hp.num_layers);

    // Create a separate ggml context + backend buffer for the cache
    size_t kv_ctx_size = ggml_tensor_overhead() * hp.num_layers * 2 + 256;
    auto * kv_ctx_buf = new uint8_t[kv_ctx_size];
    struct ggml_init_params kv_params = {
        /*.mem_size   =*/ kv_ctx_size,
        /*.mem_buffer =*/ kv_ctx_buf,
        /*.no_alloc   =*/ true,
    };
    cache.ctx = ggml_init(kv_params);

    // Create tensors in cache context
    for (int i = 0; i < hp.num_layers; i++) {
        cache.layers[i].k = ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32,
            dec_head_dim, patched_seq_len, dec_num_heads, batch_size);
        cache.layers[i].v = ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32,
            dec_head_dim, patched_seq_len, dec_num_heads, batch_size);
    }

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend_);

    // Copy computed KV data into cache
    for (int i = 0; i < hp.num_layers; i++) {
        size_t kv_size = (size_t)dec_head_dim * patched_seq_len * dec_num_heads * batch_size * sizeof(float);
        std::vector<float> tmp(dec_head_dim * patched_seq_len * dec_num_heads * batch_size);

        ggml_backend_tensor_get(k_tensors[i], tmp.data(), 0, kv_size);
        ggml_backend_tensor_set(cache.layers[i].k, tmp.data(), 0, kv_size);

        ggml_backend_tensor_get(v_tensors[i], tmp.data(), 0, kv_size);
        ggml_backend_tensor_set(cache.layers[i].v, tmp.data(), 0, kv_size);
    }

    if (std::getenv("ECHO_DEBUG_MODEL_STATS") != nullptr && !cache.layers.empty()) {
        std::string k_name = encoder_type + "_kv0_k";
        std::string v_name = encoder_type + "_kv0_v";
        print_tensor_stats(k_name.c_str(), cache.layers[0].k);
        print_tensor_stats(v_name.c_str(), cache.layers[0].v);
    }

    // Free compute context (but NOT the cache context/buffer)
    // The compute context was allocated with new[] — we need to free the backing buffer
    void * ctx_buf = ggml_get_mem_buffer(ctx);
    ggml_free(ctx);
    delete[] static_cast<uint8_t *>(ctx_buf);

    return cache;
}

EchoKVCache EchoModel::compute_text_kv_cache(
    const int32_t * input_ids,
    const float * mask,
    int seq_len,
    int batch_size
) {
    // For text: we need both IDs and mask. The build_and_compute_kv_cache currently
    // takes a single void* — we'll handle mask setting internally.
    // Pack IDs as the input, mask will be set via the graph input tensor.

    // Build the KV cache with IDs and mask
    EchoKVCache cache = build_and_compute_kv_cache("text", input_ids, mask, seq_len, batch_size);
    // The text encoder uses bidirectional attention with the provided mask.
    // TODO: The current implementation builds the mask inside the graph as an input tensor
    // that needs to be set. This is handled correctly in the graph computation.

    return cache;
}

EchoKVCache EchoModel::compute_speaker_kv_cache(
    const float * speaker_latent,
    int seq_len,
    int batch_size
) {
    return build_and_compute_kv_cache("speaker", speaker_latent, nullptr, seq_len, batch_size);
}

EchoKVCache EchoModel::compute_latent_kv_cache(
    const float * prefix_latent,
    int seq_len,
    int batch_size
) {
    return build_and_compute_kv_cache("latent", prefix_latent, nullptr, seq_len, batch_size);
}

// ────────────────────────────────────────────────────────────────────
// Decoder forward pass
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoModel::forward(
    const float * x_data,
    const float * t_data,
    int seq_len,
    int batch_size,
    const float * text_mask_data,
    int text_seq_len,
    const float * speaker_mask_data,
    int speaker_seq_len,
    const EchoKVCache & kv_text,
    const EchoKVCache & kv_speaker,
    int start_pos,
    const EchoKVCache * kv_latent
) {
    auto & hp = hparams_;
    auto & w  = weights_;
    const bool debug_stats = std::getenv("ECHO_DEBUG_MODEL_STATS") != nullptr;
    std::vector<std::pair<std::string, struct ggml_tensor *>> debug_tensors;

    // ── Compute timestep embedding on CPU ──
    // freqs = 1000 * exp(-log(10000) * arange(0, half) / half)
    // embedding = cat(cos(t * freqs), sin(t * freqs))
    int half = hp.timestep_embed_size / 2;
    std::vector<float> t_embed(batch_size * hp.timestep_embed_size);
    for (int b = 0; b < batch_size; b++) {
        float t = t_data[b];
        for (int i = 0; i < half; i++) {
            float freq = 1000.0f * expf(-logf(10000.0f) * (float)i / (float)half);
            float arg = t * freq;
            t_embed[b * hp.timestep_embed_size + i]        = cosf(arg);
            t_embed[b * hp.timestep_embed_size + half + i] = sinf(arg);
        }
    }

    std::vector<float> cond_embed_cpu((size_t)batch_size * hp.model_size * 3);
    {
        std::vector<float> h0(hp.model_size);
        std::vector<float> h1(hp.model_size);

        for (int b = 0; b < batch_size; b++) {
            const float * t_in = t_embed.data() + (size_t)b * hp.timestep_embed_size;

            for (int o = 0; o < hp.model_size; o++) {
                double sum = 0.0;
                const float * row = cond_w0_cpu_.data() + (size_t)o * hp.timestep_embed_size;
                for (int i = 0; i < hp.timestep_embed_size; i++) {
                    sum += (double)row[i] * (double)t_in[i];
                }
                h0[o] = silu_f32((float)sum);
            }

            for (int o = 0; o < hp.model_size; o++) {
                double sum = 0.0;
                const float * row = cond_w1_cpu_.data() + (size_t)o * hp.model_size;
                for (int i = 0; i < hp.model_size; i++) {
                    sum += (double)row[i] * (double)h0[i];
                }
                h1[o] = silu_f32((float)sum);
            }

            float * out = cond_embed_cpu.data() + (size_t)b * hp.model_size * 3;
            for (int o = 0; o < hp.model_size * 3; o++) {
                double sum = 0.0;
                const float * row = cond_w2_cpu_.data() + (size_t)o * hp.model_size;
                for (int i = 0; i < hp.model_size; i++) {
                    sum += (double)row[i] * (double)h1[i];
                }
                out[o] = (float)sum;
            }
        }
    }

    // ── Downsample speaker mask ──
    // speaker_mask = speaker_mask[..., ::speaker_patch_size]
    int speaker_seq_patched = speaker_seq_len / hp.speaker_patch_size;
    std::vector<float> speaker_mask_ds(batch_size * speaker_seq_patched);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < speaker_seq_patched; i++) {
            speaker_mask_ds[b * speaker_seq_patched + i] =
                speaker_mask_data[b * speaker_seq_len + i * hp.speaker_patch_size];
        }
    }

    // ── Build the forward graph ──
    size_t n_tensors = 4096;  // decoder is large
    struct ggml_context * ctx = create_compute_ctx(n_tensors);

    // Input tensors
    struct ggml_tensor * x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        hp.latent_size, seq_len, batch_size);
    ggml_set_input(x_in);
    ggml_set_name(x_in, "x_in");

    // Text mask for decoder (already float, 1.0/0.0)
    // Used only for mask data computation below, not as a graph tensor.

    // ── Cond module: timestep → conditioning ──
    // Computed on CPU above and fed as a graph input. This avoids the GGML CUDA
    // F16 matrix-vector path for cond_module.4.weight, which produced bad scale.
    struct ggml_tensor * cond_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        hp.model_size * 3, 1, batch_size);
    ggml_set_input(cond_in);
    ggml_set_name(cond_in, "cond_embed");
    struct ggml_tensor * cond = ggml_dup(ctx, cond_in);
    if (debug_stats) {
        ggml_set_output(cond);
        debug_tensors.push_back({"cond", cond});
    }

    // ── Input projection ──
    struct ggml_tensor * x = ggml_mul_mat(ctx, w.in_proj, x_in);
    x = ggml_add(ctx, x, w.in_proj_bias);
    // x: [model_size, seq, batch]
    if (debug_stats) {
        ggml_set_output(x);
        ggml_set_name(x, "debug_in_proj");
        debug_tensors.push_back({"in_proj", x});
    }

    // ── Decoder positions ──
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(pos);
    ggml_set_name(pos, "dec_pos");

    // ── Build attention masks for each decoder layer ──
    // The joint attention concatenates KV from [self, latent?, text, speaker].
    // Mask shape: [total_kv, seq_len, 1, batch]
    // - self: all 0.0 (attend)
    // - latent: 0.0 where pos < start_pos, -inf elsewhere
    // - text: 0.0 where text_mask=1.0, -inf where text_mask=0.0
    // - speaker: 0.0 where speaker_mask=1.0, -inf where speaker_mask=0.0

    int latent_kv_len = 0;
    if (kv_latent && kv_latent->layers.size() > 0) {
        latent_kv_len = kv_latent->seq_len;
    }
    int total_kv = seq_len + latent_kv_len + text_seq_len + speaker_seq_patched;

    struct ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
        total_kv, seq_len, 1, batch_size);
    ggml_set_input(attn_mask);
    ggml_set_name(attn_mask, "dec_attn_mask");

    // ── Decoder blocks ──
    for (int i = 0; i < hp.num_layers; i++) {
        const EchoKVPair * lat_kv = nullptr;
        if (kv_latent && i < (int)kv_latent->layers.size()) {
            lat_kv = &kv_latent->layers[i];
        }
        x = build_decoder_block(
            ctx, x, cond, attn_mask, pos, start_pos,
            kv_text.layers[i], kv_speaker.layers[i],
            w.decoder_blocks[i], lat_kv, i,
            debug_stats ? &debug_tensors : nullptr
        );
        if (debug_stats) {
            ggml_set_output(x);
            std::string name = "layer_" + std::to_string(i);
            ggml_set_name(x, ("debug_" + name).c_str());
            debug_tensors.push_back({name, x});
        }
    }

    // ── Output ──
    x = build_rms_norm(ctx, x, w.out_norm, hp.norm_eps);
    if (debug_stats) {
        ggml_set_output(x);
        ggml_set_name(x, "debug_out_norm");
        debug_tensors.push_back({"out_norm", x});
    }
    x = ggml_mul_mat(ctx, w.out_proj, x);
    x = ggml_add(ctx, x, w.out_proj_bias);
    // x: [latent_size=80, seq, batch]

    ggml_set_output(x);
    ggml_set_name(x, "output");

    // ── Build and compute graph ──
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        fprintf(stderr, "[echo_model] ERROR: failed to allocate forward graph\n");
        void * ctx_buf = ggml_get_mem_buffer(ctx);
        ggml_free(ctx);
        delete[] static_cast<uint8_t *>(ctx_buf);
        return {};
    }

    // ── Set input data ──
    ggml_backend_tensor_set(ggml_get_tensor(ctx, "x_in"), x_data, 0,
        (size_t)hp.latent_size * seq_len * batch_size * sizeof(float));
    ggml_backend_tensor_set(ggml_get_tensor(ctx, "cond_embed"), cond_embed_cpu.data(), 0,
        (size_t)hp.model_size * 3 * batch_size * sizeof(float));

    // Set decoder positions (start_pos offset)
    {
        std::vector<int32_t> positions(seq_len);
        for (int i = 0; i < seq_len; i++) {
            positions[i] = start_pos + i;
        }
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "dec_pos"), positions.data(), 0,
            seq_len * sizeof(int32_t));
    }

    // Build and set attention mask on CPU
    {
        std::vector<ggml_fp16_t> mask_data(total_kv * seq_len * batch_size);
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-65000.0f);
        ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);

        for (int b = 0; b < batch_size; b++) {
            for (int q = 0; q < seq_len; q++) {
                ggml_fp16_t * row = mask_data.data() + b * total_kv * seq_len + q * total_kv;
                int offset = 0;

                // Self KV: all attend
                for (int k = 0; k < seq_len; k++) {
                    row[offset + k] = zero;
                }
                offset += seq_len;

                // Latent KV: attend where position < start_pos
                if (latent_kv_len > 0) {
                    for (int k = 0; k < latent_kv_len; k++) {
                        int lat_pos = k * hp.speaker_patch_size;
                        row[offset + k] = (lat_pos < start_pos) ? zero : neg_inf;
                    }
                    offset += latent_kv_len;
                }

                // Text KV
                for (int k = 0; k < text_seq_len; k++) {
                    float m = text_mask_data[b * text_seq_len + k];
                    row[offset + k] = (m > 0.5f) ? zero : neg_inf;
                }
                offset += text_seq_len;

                // Speaker KV
                for (int k = 0; k < speaker_seq_patched; k++) {
                    float m = speaker_mask_ds[b * speaker_seq_patched + k];
                    row[offset + k] = (m > 0.5f) ? zero : neg_inf;
                }
            }
        }
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "dec_attn_mask"),
            mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    // ── Compute ──
    ggml_backend_sched_graph_compute(sched_, gf);

    if (debug_stats) {
        printf("[debug] model.forward tensor stats\n");
        for (auto & item : debug_tensors) {
            print_tensor_stats(item.first.c_str(), item.second);
        }
        print_tensor_stats("output", ggml_get_tensor(ctx, "output"));
    }

    // ── Read output ──
    size_t out_size = (size_t)hp.latent_size * seq_len * batch_size;
    std::vector<float> result(out_size);
    ggml_backend_tensor_get(ggml_get_tensor(ctx, "output"), result.data(), 0,
        out_size * sizeof(float));

    // Cleanup compute context
    void * ctx_buf = ggml_get_mem_buffer(ctx);
    ggml_free(ctx);
    delete[] static_cast<uint8_t *>(ctx_buf);

    return result;
}
