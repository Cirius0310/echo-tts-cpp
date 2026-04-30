#pragma once
// Echo-TTS C++ — Pipeline (orchestrates the full text→audio flow)
//
// Combines all components: model loading, tokenization, DAC encoding,
// PCA transform, sampling, DAC decoding, and post-processing.

#include "echo_model.h"
#include "echo_sampler.h"
#include "echo_tokenizer.h"
#include "echo_pca.h"
#include "echo_audio.h"
#include "echo_postprocess.h"

#ifdef ECHO_HAS_ONNX
#include "echo_dac_onnx.h"
#endif

#include <string>
#include <vector>

// ────────────────────────────────────────────────────────────────────
// Pipeline Configuration
// ────────────────────────────────────────────────────────────────────

struct EchoPipelineConfig {
    std::string model_path;           // GGUF file (required)
    std::string dac_encoder_path;     // ONNX (required for generation)
    std::string dac_decoder_path;     // ONNX (required for generation)
};

// ────────────────────────────────────────────────────────────────────
// Pipeline Class
// ────────────────────────────────────────────────────────────────────

class EchoPipeline {
public:
    EchoPipeline() = default;
    ~EchoPipeline() = default;

    // Load model and DAC sessions.
    bool load(const EchoPipelineConfig & config);

    // Standard generation: text + speaker WAV → output audio.
    // Returns mono float32 audio at 44100Hz.
    std::vector<float> generate(
        const std::string & text,
        const std::string & speaker_wav_path,
        const EchoSamplerConfig & sampler_config
    );

    // Blockwise generation with optional continuation.
    std::vector<float> generate_blockwise(
        const std::string & text,
        const std::string & speaker_wav_path,
        const EchoBlockwiseConfig & config,
        const std::string & continuation_wav_path = ""
    );

    // Diagnostic dump: save intermediates to directory for comparison with Python.
    // Dumps: speaker_zq, speaker_zpca, speaker_mask, token_ids, x_t, model_fwd_out
    // Each gets a .bin (raw data) + .shape (dimensions, Python batch-first order).
    bool diagnostic_dump(
        const std::string & out_dir,
        const std::string & text,
        const std::string & speaker_wav_path,
        const EchoSamplerConfig & config
    );

    // Access internals
    EchoModel & model() { return model_; }
    const EchoModel & model() const { return model_; }

private:
    EchoModel model_;
#ifdef ECHO_HAS_ONNX
    EchoDACSession dac_;
#endif

    // Helper: encode speaker audio to PCA latent + mask
    struct SpeakerData {
        std::vector<float> latent;    // (seq_len, 80)
        std::vector<float> mask;      // (seq_len,)
        int seq_len = 0;
    };
    SpeakerData encode_speaker(const std::string & wav_path);
};
