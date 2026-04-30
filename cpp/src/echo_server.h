#pragma once
// Echo-TTS C++ — OpenAI-compatible HTTP API Server
//
// Serves POST /v1/audio/speech with OpenAI TTS-compatible request/response.
// Voices are pre-encoded on startup (DAC+PCA cached) for low-latency generation.

#include "echo_pipeline.h"
#include "echo_sampler.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// ────────────────────────────────────────────────────────────────────
// Server Configuration
// ────────────────────────────────────────────────────────────────────

struct EchoServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;

    std::string model_path;
    std::string dac_encoder_path;
    std::string dac_decoder_path;

    // Voice name → speaker WAV file path
    std::unordered_map<std::string, std::string> voices;

    // Default sampling parameters (overridable per-request?)
    EchoSamplerConfig sampler_defaults;
};

// ────────────────────────────────────────────────────────────────────
// Server Class
// ────────────────────────────────────────────────────────────────────

class EchoServer {
public:
    EchoServer() = default;
    ~EchoServer() = default;

    // Load model, pre-encode voices, start HTTP listener. Blocks until shutdown.
    bool start(const EchoServerConfig & config);

    // Signal the server to stop (thread-safe).
    void stop();

private:
    EchoPipeline pipeline_;
    EchoSamplerConfig sampler_defaults_;
    std::mutex gpu_mutex_;

    // Pre-cached speaker data: voice name → PCA latent + mask
    std::unordered_map<std::string, SpeakerLatentData> voice_cache_;

    // Available model names
    std::string model_name_ = "echo-tts";

    // Internal helpers
    bool pre_encode_voices(const std::unordered_map<std::string, std::string> & voices);
    static std::vector<uint8_t> audio_to_wav_bytes(const std::vector<float> & audio, int sample_rate);
    static std::vector<uint8_t> audio_to_pcm_bytes(const std::vector<float> & audio);
    static std::vector<uint8_t> audio_to_mp3_bytes(const std::vector<float> & audio, int sample_rate);
    static std::string json_error(const std::string & message, const std::string & type,
                                   const std::string & param, const std::string & code);
    static bool check_ffmpeg_available();

    bool ffmpeg_available_ = false;
};
