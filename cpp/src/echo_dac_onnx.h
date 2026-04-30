#pragma once
// Echo-TTS C++ — S1-DAC ONNX Runtime Wrapper
//
// Wraps the ONNX-exported S1-DAC encoder and decoder for audio↔latent conversion.
// Conditionally compiled only when ECHO_HAS_ONNX is defined.

#ifdef ECHO_HAS_ONNX

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class EchoDACSession {
public:
    EchoDACSession();
    ~EchoDACSession();

    // Load ONNX models. Returns false on failure.
    bool load(const std::string & encoder_path, const std::string & decoder_path);

    // Encode audio waveform → z_q (pre-PCA continuous latent).
    // audio: (batch, 1, length) — mono float32 at 44100Hz
    // Returns z_q: (batch, 1024, time)
    std::vector<float> encode(
        const float * audio,
        int batch_size,
        int audio_length,
        int & out_time_steps
    );

    // Decode z_q → audio waveform.
    // z_q: (batch, 1024, time) — continuous latent
    // Returns audio: (batch, 1, length)
    std::vector<float> decode(
        const float * z_q,
        int batch_size,
        int time_steps,
        int & out_audio_length
    );

    bool is_loaded() const { return encoder_session_ != nullptr && decoder_session_ != nullptr; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    Ort::MemoryInfo memory_info_;
};

#endif // ECHO_HAS_ONNX
