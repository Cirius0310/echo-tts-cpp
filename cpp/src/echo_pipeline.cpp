// Echo-TTS C++ — Pipeline Implementation

#include "echo_pipeline.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>

#ifdef _WIN32
#include <direct.h>
#define mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define mkdir(p) mkdir(p, 0755)
#endif

// ────────────────────────────────────────────────────────────────────
// Load
// ────────────────────────────────────────────────────────────────────

bool EchoPipeline::load(const EchoPipelineConfig & config) {
    printf("[pipeline] Loading model...\n");
    if (!model_.load(config.model_path)) {
        fprintf(stderr, "[pipeline] Failed to load model\n");
        return false;
    }

#ifdef ECHO_HAS_ONNX
    if (!config.dac_encoder_path.empty() && !config.dac_decoder_path.empty()) {
        printf("[pipeline] Loading DAC sessions...\n");
        if (!dac_.load(config.dac_encoder_path, config.dac_decoder_path)) {
            fprintf(stderr, "[pipeline] Failed to load DAC ONNX models\n");
            return false;
        }
    }
#else
    if (!config.dac_encoder_path.empty()) {
        fprintf(stderr, "[pipeline] WARNING: ONNX support not compiled. DAC sessions not loaded.\n");
    }
#endif

    printf("[pipeline] Ready.\n");
    return true;
}

// ────────────────────────────────────────────────────────────────────
// Speaker encoding helper
// ────────────────────────────────────────────────────────────────────

SpeakerLatentData EchoPipeline::encode_speaker(const std::string & wav_path) {
    SpeakerLatentData data;
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    // Load audio
    std::vector<float> audio = load_wav(wav_path, 44100);
    if (audio.empty()) {
        fprintf(stderr, "[pipeline] Failed to load speaker audio: %s\n", wav_path.c_str());
        return data;
    }

    printf("[pipeline] Speaker audio: %zu samples (%.1f sec)\n",
           audio.size(), audio.size() / 44100.0f);

#ifdef ECHO_HAS_ONNX
    if (!dac_.is_loaded()) {
        fprintf(stderr, "[pipeline] ERROR: DAC not loaded, cannot encode speaker\n");
        return data;
    }

    // Encode audio → z_q via DAC
    // DAC expects (batch=1, channels=1, length)
    int audio_len = (int)audio.size();

    // Process in chunks (matching Python: 640 * 2048 = ~30s chunks)
    int chunk_size = 640 * AE_DOWNSAMPLE_FACTOR;
    int max_latent_len = 6400;  // max speaker latent length from training
    int max_audio_len = max_latent_len * AE_DOWNSAMPLE_FACTOR;
    audio_len = std::min(audio_len, max_audio_len);

    std::vector<float> all_latent;
    int total_latent_frames = 0;

    for (int offset = 0; offset < audio_len; offset += chunk_size) {
        int this_chunk = std::min(chunk_size, audio_len - offset);

        // Pad chunk to chunk_size if shorter
        std::vector<float> chunk(chunk_size, 0.0f);
        std::copy(audio.data() + offset, audio.data() + offset + this_chunk, chunk.data());

        int time_steps = 0;
        std::vector<float> z_q = dac_.encode(chunk.data(), 1, chunk_size, time_steps);

        // PCA encode: z_q (1, 1024, T) → z_pca (1, T, 80)
        std::vector<float> z_pca(time_steps * hp.latent_size);
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        pca_encode(pca_params, z_q.data(), z_pca.data(), 1, time_steps);

        all_latent.insert(all_latent.end(), z_pca.begin(), z_pca.end());
        total_latent_frames += time_steps;
    }

    // Compute actual latent length (before padding)
    int actual_latent_len = audio_len / AE_DOWNSAMPLE_FACTOR;

    // Trim to actual length (don't pad to max)
    actual_latent_len = std::min(actual_latent_len, total_latent_frames);

    // Ensure divisible by patch_size
    actual_latent_len = (actual_latent_len / hp.speaker_patch_size) * hp.speaker_patch_size;

    data.latent.resize(actual_latent_len * hp.latent_size);
    std::copy(all_latent.begin(),
              all_latent.begin() + actual_latent_len * hp.latent_size,
              data.latent.data());

    data.mask.resize(actual_latent_len, 1.0f);
    data.seq_len = actual_latent_len;
#else
    fprintf(stderr, "[pipeline] ERROR: ONNX not available, cannot encode speaker\n");
#endif

    printf("[pipeline] Speaker latent: %d frames\n", data.seq_len);
    return data;
}

// ────────────────────────────────────────────────────────────────────
// Standard generation
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoPipeline::generate(
    const std::string & text,
    const std::string & speaker_wav_path,
    const EchoSamplerConfig & sampler_config
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    // ── Tokenize ──
    printf("[pipeline] Tokenizing text...\n");
    EchoTokenizerResult tok = get_text_input_ids_and_mask(text);
    printf("[pipeline] Text: \"%s\" → %d tokens\n",
           tok.normalized_text.c_str(), tok.actual_length);

    // Convert mask to float
    std::vector<float> text_mask_f(tok.mask.size());
    for (size_t i = 0; i < tok.mask.size(); i++) {
        text_mask_f[i] = tok.mask[i] ? 1.0f : 0.0f;
    }

    // ── Encode speaker ──
    printf("[pipeline] Encoding speaker audio...\n");
    SpeakerLatentData speaker = encode_speaker(speaker_wav_path);
    if (speaker.seq_len == 0) {
        // No speaker: use zero latent (unconditional speaker)
        int dummy_len = hp.speaker_patch_size;
        speaker.latent.resize(dummy_len * hp.latent_size, 0.0f);
        speaker.mask.resize(dummy_len, 0.0f);
        speaker.seq_len = dummy_len;
    }

    // ── Sample ──
    printf("[pipeline] Sampling...\n");
    EchoSamplerResult sampled = sample_euler_cfg(
        model_, sampler_config,
        speaker.latent.data(), speaker.mask.data(), speaker.seq_len,
        tok.token_ids.data(), text_mask_f.data(), (int)tok.token_ids.size()
    );

    // ── Decode ──
    printf("[pipeline] Decoding latent → audio...\n");
    std::vector<float> output_audio;

#ifdef ECHO_HAS_ONNX
    if (dac_.is_loaded()) {
        // PCA decode: z_pca (1, T, 80) → z_q (1, 1024, T)
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        std::vector<float> z_q(sampled.seq_len * 1024);
        pca_decode(pca_params, sampled.latent.data(), z_q.data(), 1, sampled.seq_len);

        // DAC decode: z_q (1, 1024, T) → audio (1, 1, L)
        int out_audio_len = 0;
        output_audio = dac_.decode(z_q.data(), 1, sampled.seq_len, out_audio_len);

        // Crop at flattening point
        int crop_len = crop_length_from_latent(
            sampled.latent.data(), sampled.seq_len, hp.latent_size
        );
        if (crop_len > 0 && crop_len < out_audio_len) {
            output_audio.resize(crop_len);
        }

        printf("[pipeline] Output: %zu samples (%.1f sec)\n",
               output_audio.size(), output_audio.size() / 44100.0f);
    }
#else
    fprintf(stderr, "[pipeline] WARNING: No DAC decoder. Returning raw latent.\n");
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[pipeline] Total generation time: %.1f seconds\n", elapsed);

    return output_audio;
}

// ────────────────────────────────────────────────────────────────────
// Generation from pre-encoded speaker latent
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoPipeline::generate_from_latent(
    const std::string & text,
    const SpeakerLatentData & speaker,
    const EchoSamplerConfig & sampler_config
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    // ── Tokenize ──
    printf("[pipeline] Tokenizing text...\n");
    EchoTokenizerResult tok = get_text_input_ids_and_mask(text);
    printf("[pipeline] Text: \"%s\" → %d tokens\n",
           tok.normalized_text.c_str(), tok.actual_length);

    // Convert mask to float
    std::vector<float> text_mask_f(tok.mask.size());
    for (size_t i = 0; i < tok.mask.size(); i++) {
        text_mask_f[i] = tok.mask[i] ? 1.0f : 0.0f;
    }

    // ── Validate speaker ──
    SpeakerLatentData spk = speaker;
    if (spk.seq_len == 0) {
        int dummy_len = hp.speaker_patch_size;
        spk.latent.resize(dummy_len * hp.latent_size, 0.0f);
        spk.mask.resize(dummy_len, 0.0f);
        spk.seq_len = dummy_len;
    }

    // ── Sample ──
    printf("[pipeline] Sampling...\n");
    EchoSamplerResult sampled = sample_euler_cfg(
        model_, sampler_config,
        spk.latent.data(), spk.mask.data(), spk.seq_len,
        tok.token_ids.data(), text_mask_f.data(), (int)tok.token_ids.size()
    );

    // ── Decode ──
    printf("[pipeline] Decoding latent → audio...\n");
    std::vector<float> output_audio;

#ifdef ECHO_HAS_ONNX
    if (dac_.is_loaded()) {
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        std::vector<float> z_q(sampled.seq_len * 1024);
        pca_decode(pca_params, sampled.latent.data(), z_q.data(), 1, sampled.seq_len);

        int out_audio_len = 0;
        output_audio = dac_.decode(z_q.data(), 1, sampled.seq_len, out_audio_len);

        int crop_len = crop_length_from_latent(
            sampled.latent.data(), sampled.seq_len, hp.latent_size
        );
        if (crop_len > 0 && crop_len < out_audio_len) {
            output_audio.resize(crop_len);
        }

        printf("[pipeline] Output: %zu samples (%.1f sec)\n",
               output_audio.size(), output_audio.size() / 44100.0f);
    }
#else
    fprintf(stderr, "[pipeline] WARNING: No DAC decoder. Returning raw latent.\n");
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[pipeline] Total generation time: %.1f seconds\n", elapsed);

    return output_audio;
}

// ────────────────────────────────────────────────────────────────────
// Speaker KV cache pre-computation (for reuse across chunks)
// ────────────────────────────────────────────────────────────────────

EchoKVCache EchoPipeline::compute_speaker_kv(const SpeakerLatentData & speaker) {
    auto & hp = model_.hparams();
    SpeakerLatentData spk = speaker;
    if (spk.seq_len == 0) {
        int dummy_len = hp.speaker_patch_size;
        spk.latent.resize(dummy_len * hp.latent_size, 0.0f);
        spk.mask.resize(dummy_len, 0.0f);
        spk.seq_len = dummy_len;
    }
    return model_.compute_speaker_kv_cache(spk.latent.data(), spk.seq_len, 1);
}

// ────────────────────────────────────────────────────────────────────
// Generation from pre-encoded speaker latent with pre-computed speaker KV
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoPipeline::generate_from_latent_with_speaker_kv(
    const std::string & text,
    const SpeakerLatentData & speaker,
    const EchoSamplerConfig & sampler_config,
    const EchoKVCache & kv_speaker
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    printf("[pipeline] Tokenizing text...\n");
    EchoTokenizerResult tok = get_text_input_ids_and_mask(text);
    printf("[pipeline] Text: \"%s\" → %d tokens\n",
           tok.normalized_text.c_str(), tok.actual_length);

    std::vector<float> text_mask_f(tok.mask.size());
    for (size_t i = 0; i < tok.mask.size(); i++) {
        text_mask_f[i] = tok.mask[i] ? 1.0f : 0.0f;
    }

    SpeakerLatentData spk = speaker;
    if (spk.seq_len == 0) {
        int dummy_len = hp.speaker_patch_size;
        spk.latent.resize(dummy_len * hp.latent_size, 0.0f);
        spk.mask.resize(dummy_len, 0.0f);
        spk.seq_len = dummy_len;
    }

    printf("[pipeline] Sampling (with pre-computed speaker KV)...\n");
    EchoSamplerResult sampled = sample_euler_cfg_with_speaker_kv(
        model_, sampler_config,
        spk.mask.data(), spk.seq_len,
        tok.token_ids.data(), text_mask_f.data(), (int)tok.token_ids.size(),
        kv_speaker
    );

    printf("[pipeline] Decoding latent → audio...\n");
    std::vector<float> output_audio;

#ifdef ECHO_HAS_ONNX
    if (dac_.is_loaded()) {
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        std::vector<float> z_q(sampled.seq_len * 1024);
        pca_decode(pca_params, sampled.latent.data(), z_q.data(), 1, sampled.seq_len);

        int out_audio_len = 0;
        output_audio = dac_.decode(z_q.data(), 1, sampled.seq_len, out_audio_len);

        int crop_len = crop_length_from_latent(
            sampled.latent.data(), sampled.seq_len, hp.latent_size
        );
        if (crop_len > 0 && crop_len < out_audio_len) {
            output_audio.resize(crop_len);
        }

        printf("[pipeline] Output: %zu samples (%.1f sec)\n",
               output_audio.size(), output_audio.size() / 44100.0f);
    }
#else
    fprintf(stderr, "[pipeline] WARNING: No DAC decoder. Returning raw latent.\n");
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[pipeline] Total generation time: %.1f seconds\n", elapsed);

    return output_audio;
}

// ────────────────────────────────────────────────────────────────────
// Blockwise generation
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoPipeline::generate_blockwise(
    const std::string & text,
    const std::string & speaker_wav_path,
    const EchoBlockwiseConfig & config,
    const std::string & continuation_wav_path
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    // Tokenize
    EchoTokenizerResult tok = get_text_input_ids_and_mask(text);
    std::vector<float> text_mask_f(tok.mask.size());
    for (size_t i = 0; i < tok.mask.size(); i++) {
        text_mask_f[i] = tok.mask[i] ? 1.0f : 0.0f;
    }

    // Encode speaker
    SpeakerLatentData speaker = encode_speaker(speaker_wav_path);
    if (speaker.seq_len == 0) {
        int dummy_len = hp.speaker_patch_size;
        speaker.latent.resize(dummy_len * hp.latent_size, 0.0f);
        speaker.mask.resize(dummy_len, 0.0f);
        speaker.seq_len = dummy_len;
    }

    // Encode continuation (if provided)
    const float * cont_latent = nullptr;
    int cont_len = 0;
    SpeakerLatentData continuation;
    if (!continuation_wav_path.empty()) {
        printf("[pipeline] Encoding continuation audio...\n");
        continuation = encode_speaker(continuation_wav_path);
        cont_latent = continuation.latent.data();
        cont_len = continuation.seq_len;
    }

    // Sample blockwise
    EchoSamplerResult sampled = sample_blockwise(
        model_, config,
        speaker.latent.data(), speaker.mask.data(), speaker.seq_len,
        tok.token_ids.data(), text_mask_f.data(), (int)tok.token_ids.size(),
        cont_latent, cont_len
    );

    // Decode
    std::vector<float> output_audio;

#ifdef ECHO_HAS_ONNX
    if (dac_.is_loaded()) {
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        std::vector<float> z_q(sampled.seq_len * 1024);
        pca_decode(pca_params, sampled.latent.data(), z_q.data(), 1, sampled.seq_len);

        int out_audio_len = 0;
        output_audio = dac_.decode(z_q.data(), 1, sampled.seq_len, out_audio_len);

        int crop_len = crop_length_from_latent(
            sampled.latent.data(), sampled.seq_len, hp.latent_size
        );
        if (crop_len > 0 && crop_len < out_audio_len) {
            output_audio.resize(crop_len);
        }

        printf("[pipeline] Output: %zu samples (%.1f sec)\n",
               output_audio.size(), output_audio.size() / 44100.0f);
    }
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();
    printf("[pipeline] Total blockwise generation time: %.1f seconds\n", elapsed);

    return output_audio;
}

// ────────────────────────────────────────────────────────────────────
// Diagnostic dump helpers
// ────────────────────────────────────────────────────────────────────

// Write raw binary data with a companion .shape file (dimensions in Python batch-first order).
static void write_binary_dump(
    const std::string & base_path,
    const float * data, size_t count,
    const std::vector<int> & shape  // Python order: batch, seq, feature
) {
    // Write .bin
    {
        std::ofstream f(base_path + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    }
    // Write .shape
    {
        std::ofstream f(base_path + ".shape");
        for (size_t i = 0; i < shape.size(); i++) {
            f << shape[i] << "\n";
        }
    }
    // Verify size matches
    size_t expected = 1;
    for (int s : shape) expected *= (size_t)s;
    if (expected != count) {
        fprintf(stderr, "[dump] WARNING: %s: shape product %zu != count %zu\n",
                base_path.c_str(), expected, count);
    }
    printf("[dump] %s: shape [", base_path.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        printf("%d%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    printf("] = %zu floats\n", count);
}

static void write_binary_dump_i32(
    const std::string & base_path,
    const int32_t * data, size_t count,
    const std::vector<int> & shape
) {
    {
        std::ofstream f(base_path + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(data), count * sizeof(int32_t));
    }
    {
        std::ofstream f(base_path + ".shape");
        f << "int32\n";
        for (size_t i = 0; i < shape.size(); i++) {
            f << shape[i] << "\n";
        }
    }
    printf("[dump] %s: shape [", base_path.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        printf("%d%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    printf("] = %zu int32\n", count);
}

// ────────────────────────────────────────────────────────────────────
// Diagnostic dump
// ────────────────────────────────────────────────────────────────────

bool EchoPipeline::diagnostic_dump(
    const std::string & out_dir,
    const std::string & text,
    const std::string & speaker_wav_path,
    const EchoSamplerConfig & config
) {
    auto & hp = model_.hparams();
    auto & pca = model_.pca_state();

    // Create output directory
    mkdir(out_dir.c_str());

    // ── Tokenize ──
    printf("[dump] Tokenizing text: \"%s\"\n", text.c_str());
    EchoTokenizerResult tok = get_text_input_ids_and_mask(text);
    printf("[dump] Normalized: \"%s\" → %d tokens\n",
           tok.normalized_text.c_str(), tok.actual_length);

    // Dump token_ids: Python shape (1, text_len) int32
    write_binary_dump_i32(out_dir + "/token_ids",
        tok.token_ids.data(), tok.token_ids.size(),
        {1, (int)tok.token_ids.size()});

    // ── Load and encode speaker ──
    printf("[dump] Loading speaker audio: %s\n", speaker_wav_path.c_str());
    std::vector<float> audio = load_wav(speaker_wav_path, 44100);
    if (audio.empty()) {
        fprintf(stderr, "[dump] ERROR: Failed to load speaker audio\n");
        return false;
    }

#ifdef ECHO_HAS_ONNX
    if (!dac_.is_loaded()) {
        fprintf(stderr, "[dump] ERROR: DAC not loaded\n");
        return false;
    }

    // Encode in chunks (matching Python chunking)
    int chunk_size = 640 * AE_DOWNSAMPLE_FACTOR;
    int max_latent_len = 6400;
    int max_audio_len = max_latent_len * AE_DOWNSAMPLE_FACTOR;
    int audio_len = std::min((int)audio.size(), max_audio_len);

    std::vector<float> all_zq_flat;  // flatten all chunks for dump
    std::vector<float> all_z_pca;
    int total_latent_frames = 0;

    for (int offset = 0; offset < audio_len; offset += chunk_size) {
        int this_chunk = std::min(chunk_size, audio_len - offset);
        std::vector<float> chunk(chunk_size, 0.0f);
        std::copy(audio.data() + offset, audio.data() + offset + this_chunk, chunk.data());

        int time_steps = 0;
        std::vector<float> z_q = dac_.encode(chunk.data(), 1, chunk_size, time_steps);
        // z_q shape: (1, 1024, time_steps) from ONNX — already in row-major Python order
        all_zq_flat.insert(all_zq_flat.end(), z_q.begin(), z_q.end());

        // PCA encode
        EchoPCAParams pca_params = {
            pca.components.data(), pca.mean.data(), pca.latent_scale,
            hp.latent_size, 1024
        };
        std::vector<float> z_pca(time_steps * hp.latent_size);
        pca_encode(pca_params, z_q.data(), z_pca.data(), 1, time_steps);
        // z_pca layout: (1, T, 80) in PCA order, stored flat as [b=0, t=0..T-1, p=0..79]
        all_z_pca.insert(all_z_pca.end(), z_pca.begin(), z_pca.end());
        total_latent_frames += time_steps;
    }

    // Trim to actual length
    int actual_latent_len = audio_len / AE_DOWNSAMPLE_FACTOR;
    actual_latent_len = std::min(actual_latent_len, total_latent_frames);
    actual_latent_len = (actual_latent_len / hp.speaker_patch_size) * hp.speaker_patch_size;

    // Dump speaker_zq: (1, 1024, total_time_steps)
    int total_zq_time = total_latent_frames;
    write_binary_dump(out_dir + "/speaker_zq",
        all_zq_flat.data(), all_zq_flat.size(),
        {1, 1024, total_zq_time});

    // Dump speaker_zpca: (1, actual_latent_len, 80)
    write_binary_dump(out_dir + "/speaker_zpca",
        all_z_pca.data(), (size_t)actual_latent_len * 80,
        {1, actual_latent_len, 80});

    // Speaker mask (1, actual_latent_len)
    std::vector<float> speaker_mask(actual_latent_len, 1.0f);
    write_binary_dump(out_dir + "/speaker_mask",
        speaker_mask.data(), (size_t)actual_latent_len,
        {1, actual_latent_len});

    // Build speaker latent + mask for downstream use
    std::vector<float> speaker_latent(actual_latent_len * hp.latent_size);
    std::copy(all_z_pca.begin(),
              all_z_pca.begin() + actual_latent_len * hp.latent_size,
              speaker_latent.data());

    // ── Convert mask to float ──
    std::vector<float> text_mask_f(tok.mask.size());
    for (size_t i = 0; i < tok.mask.size(); i++) {
        text_mask_f[i] = tok.mask[i] ? 1.0f : 0.0f;
    }

    // ── Compute KV caches ──
    printf("[dump] Computing KV caches...\n");
    EchoKVCache kv_text = model_.compute_text_kv_cache(
        tok.token_ids.data(), text_mask_f.data(), (int)tok.token_ids.size(), 1);
    EchoKVCache kv_speaker = model_.compute_speaker_kv_cache(
        speaker_latent.data(), actual_latent_len, 1);

    // ── Generate x_t (same seed matching Python) ──
    int seq_len = config.sequence_length;
    int batch_size = 1;
    int latent_dim = hp.latent_size;
    int total_latent = batch_size * seq_len * latent_dim;

    printf("[dump] Generating x_t (seed=%llu, seq_len=%d)...\n",
           (unsigned long long)config.rng_seed, seq_len);

    std::mt19937_64 rng(config.rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x_t(total_latent);
    for (int i = 0; i < total_latent; i++) {
        x_t[i] = normal(rng);
    }

    // Note: no truncation applied (matches what C++ sampler does by default)

    // Dump x_t in Python format: (1, seq_len, 80)
    // C++ layout: [80, seq_len, 1] GGML → rearrange to (batch=1, seq, 80)
    {
        std::vector<float> x_t_py(total_latent);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < latent_dim; d++) {
                x_t_py[s * latent_dim + d] = x_t[s * latent_dim + d];
            }
        }
        write_binary_dump(out_dir + "/x_t",
            x_t_py.data(), (size_t)total_latent,
            {1, seq_len, latent_dim});
    }

    // ── Single forward pass ──
    float t_val = 0.5f;
    printf("[dump] Running single forward pass at t=%.3f...\n", t_val);

    std::vector<float> t_vec(batch_size, t_val);
    std::vector<float> output = model_.forward(
        x_t.data(), t_vec.data(), seq_len, batch_size,
        text_mask_f.data(), (int)tok.token_ids.size(),
        speaker_mask.data(), actual_latent_len,
        kv_text, kv_speaker
    );

    // Dump model_fwd_out in Python format: (1, seq_len, 80)
    {
        std::vector<float> out_py(total_latent);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < latent_dim; d++) {
                out_py[s * latent_dim + d] = output[s * latent_dim + d];
            }
        }
        write_binary_dump(out_dir + "/model_fwd_out",
            out_py.data(), (size_t)total_latent,
            {1, seq_len, latent_dim});

        // Stats
        float out_min = out_py[0], out_max = out_py[0];
        double out_sum = 0.0;
        for (size_t i = 0; i < total_latent; i++) {
            out_min = std::min(out_min, out_py[i]);
            out_max = std::max(out_max, out_py[i]);
            out_sum += out_py[i];
        }
        printf("[dump] Output range: [%.4f, %.4f], mean=%.4f\n",
               out_min, out_max, (float)(out_sum / total_latent));
    }

    printf("[dump] Done! All intermediates saved to %s/\n", out_dir.c_str());
    return true;
#else
    fprintf(stderr, "[dump] ERROR: ONNX support not compiled, cannot run diagnostic dump\n");
    return false;
#endif
}
