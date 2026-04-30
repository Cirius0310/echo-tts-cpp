// Echo-TTS C++ — Server Implementation

#include "echo_server.h"
#include "echo_audio.h"

#include <httplib.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::json;

// ────────────────────────────────────────────────────────────────────
// Global pointer for signal handler
// ────────────────────────────────────────────────────────────────────

static httplib::Server * g_http_server = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_http_server) {
        g_http_server->stop();
    }
}

// ────────────────────────────────────────────────────────────────────
// Audio serialization helpers
// ────────────────────────────────────────────────────────────────────

std::vector<uint8_t> EchoServer::audio_to_wav_bytes(const std::vector<float> & audio, int sample_rate) {
    int num_samples = (int)audio.size();
    std::vector<uint8_t> wav(44 + num_samples * 2);
    uint8_t * p = wav.data();

    // Clamp and convert float → int16
    std::vector<int16_t> samples(num_samples);
    for (int i = 0; i < num_samples; i++) {
        float clamped = std::max(-1.0f, std::min(1.0f, audio[i]));
        samples[i] = (int16_t)(clamped * 32767.0f);
    }

    uint32_t data_size = (uint32_t)(num_samples * 2);
    uint32_t file_size = 36 + data_size;

    // RIFF header
    memcpy(p, "RIFF", 4); p += 4;
    memcpy(p, &file_size, 4); p += 4;
    memcpy(p, "WAVE", 4); p += 4;

    // fmt chunk
    memcpy(p, "fmt ", 4); p += 4;
    uint32_t fmt_size = 16;
    memcpy(p, &fmt_size, 4); p += 4;
    uint16_t audio_format = 1;
    memcpy(p, &audio_format, 2); p += 2;
    uint16_t num_channels = 1;
    memcpy(p, &num_channels, 2); p += 2;
    uint32_t sr = (uint32_t)sample_rate;
    memcpy(p, &sr, 4); p += 4;
    uint32_t byte_rate = sr * num_channels * 2;
    memcpy(p, &byte_rate, 4); p += 4;
    uint16_t block_align = num_channels * 2;
    memcpy(p, &block_align, 2); p += 2;
    uint16_t bits_per_sample = 16;
    memcpy(p, &bits_per_sample, 2); p += 2;

    // data chunk
    memcpy(p, "data", 4); p += 4;
    memcpy(p, &data_size, 4); p += 4;
    memcpy(p, samples.data(), data_size);

    return wav;
}

std::vector<uint8_t> EchoServer::audio_to_pcm_bytes(const std::vector<float> & audio) {
    int num_samples = (int)audio.size();
    std::vector<uint8_t> pcm(num_samples * 2);
    int16_t * out = (int16_t *)pcm.data();
    for (int i = 0; i < num_samples; i++) {
        float clamped = std::max(-1.0f, std::min(1.0f, audio[i]));
        out[i] = (int16_t)(clamped * 32767.0f);
    }
    return pcm;
}

bool EchoServer::check_ffmpeg_available() {
#ifdef _WIN32
    int ret = system("ffmpeg -version >nul 2>nul");
#else
    int ret = system("ffmpeg -version >/dev/null 2>/dev/null");
#endif
    return ret == 0;
}

std::vector<uint8_t> EchoServer::audio_to_mp3_bytes(const std::vector<float> & audio, int sample_rate) {
    if (!check_ffmpeg_available()) {
        fprintf(stderr, "[server] ERROR: ffmpeg not found on PATH (required for MP3 encoding)\n");
        return {};
    }

    // Temp file paths
#ifdef _WIN32
    std::string tmp_dir = getenv("TEMP") ? getenv("TEMP") : ".";
    std::string wav_path = tmp_dir + "\\echo-tts-" + std::to_string(GetCurrentProcessId()) + ".wav";
    std::string mp3_path = tmp_dir + "\\echo-tts-" + std::to_string(GetCurrentProcessId()) + ".mp3";
#else
    std::string wav_path = "/tmp/echo-tts-" + std::to_string(getpid()) + ".wav";
    std::string mp3_path = "/tmp/echo-tts-" + std::to_string(getpid()) + ".mp3";
#endif

    // Write WAV to temp file
    auto wav_bytes = audio_to_wav_bytes(audio, sample_rate);
    {
        FILE * f = fopen(wav_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[server] ERROR: Failed to write temp WAV file\n");
            return {};
        }
        fwrite(wav_bytes.data(), 1, wav_bytes.size(), f);
        fclose(f);
    }

    // Run ffmpeg
#ifdef _WIN32
    std::string cmd = "ffmpeg -y -i \"" + wav_path + "\" -codec:a libmp3lame -b:a 128k -f mp3 \"" + mp3_path + "\" 2>nul";
#else
    std::string cmd = "ffmpeg -y -i \"" + wav_path + "\" -codec:a libmp3lame -b:a 128k -f mp3 \"" + mp3_path + "\" 2>/dev/null";
#endif
    int ret = system(cmd.c_str());

    // Read MP3 result
    std::vector<uint8_t> mp3_data;
    if (ret == 0) {
        FILE * f = fopen(mp3_path.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            mp3_data.resize(size);
            fread(mp3_data.data(), 1, size, f);
            fclose(f);
        }
    } else {
        fprintf(stderr, "[server] ERROR: ffmpeg MP3 encoding failed (exit code %d)\n", ret);
    }

    // Clean up temp files
    remove(wav_path.c_str());
    remove(mp3_path.c_str());

    return mp3_data;
}

// ────────────────────────────────────────────────────────────────────
// JSON error response builder (OpenAI style)
// ────────────────────────────────────────────────────────────────────

std::string EchoServer::json_error(const std::string & message, const std::string & type,
                                     const std::string & param, const std::string & code) {
    json j;
    j["error"]["message"] = message;
    j["error"]["type"]    = type;
    if (!param.empty()) j["error"]["param"] = param;
    if (!code.empty())   j["error"]["code"]  = code;
    return j.dump();
}

// ────────────────────────────────────────────────────────────────────
// Voice pre-encoding
// ────────────────────────────────────────────────────────────────────

bool EchoServer::pre_encode_voices(const std::unordered_map<std::string, std::string> & voices) {
    printf("[server] Pre-encoding %zu voice(s)...\n", voices.size());
    for (const auto & kv : voices) {
        printf("[server]   Encoding voice '%s' from %s\n", kv.first.c_str(), kv.second.c_str());
        SpeakerLatentData data = pipeline_.encode_speaker(kv.second);
        if (data.seq_len == 0) {
            fprintf(stderr, "[server] ERROR: Failed to encode voice '%s'\n", kv.first.c_str());
            return false;
        }
        voice_cache_[kv.first] = std::move(data);
    }
    printf("[server] All voices pre-encoded.\n");
    return true;
}

// ────────────────────────────────────────────────────────────────────
// Start / Stop
// ────────────────────────────────────────────────────────────────────

bool EchoServer::start(const EchoServerConfig & config) {
    sampler_defaults_ = config.sampler_defaults;
    model_name_ = "echo-tts";

    // ── Load pipeline ──
    EchoPipelineConfig pipeline_config;
    pipeline_config.model_path       = config.model_path;
    pipeline_config.dac_encoder_path = config.dac_encoder_path;
    pipeline_config.dac_decoder_path = config.dac_decoder_path;

    if (!pipeline_.load(pipeline_config)) {
        fprintf(stderr, "[server] ERROR: Failed to load pipeline\n");
        return false;
    }

    // ── Pre-encode voices ──
    if (!pre_encode_voices(config.voices)) {
        fprintf(stderr, "[server] ERROR: Failed to pre-encode voices\n");
        return false;
    }

    // ── Check ffmpeg availability (for MP3 support) ──
    ffmpeg_available_ = check_ffmpeg_available();
    if (!ffmpeg_available_) {
        printf("[server] Note: ffmpeg not found on PATH. MP3 format unavailable.\n");
    }

    // ── Build HTTP server ──
    httplib::Server srv;
    g_http_server = &srv;

    // Set longer timeouts for TTS generation
    srv.set_read_timeout(10, 0);       // 10 seconds for reading request
    srv.set_write_timeout(120, 0);     // 2 minutes for writing response
    srv.set_idle_interval(0, 100000);  // 100ms keep-alive polling

    // ── CORS middleware (allow all origins) ──
    srv.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    // ── GET /health ──
    srv.Get("/health", [this](const httplib::Request &, httplib::Response & res) {
        json j;
        j["status"] = "ok";
        j["model"]  = model_name_;
        j["voices"] = json::array();
        for (const auto & kv : voice_cache_) {
            j["voices"].push_back(kv.first);
        }
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(j.dump(), "application/json");
    });

    // ── GET /v1/audio/models ──
    srv.Get("/v1/audio/models", [this](const httplib::Request &, httplib::Response & res) {
        json j;
        j["object"] = "list";
        j["data"]   = json::array();

        json model_entry;
        model_entry["id"]       = model_name_;
        model_entry["object"]   = "model";
        model_entry["owned_by"] = "echo-tts";
        j["data"].push_back(model_entry);

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(j.dump(), "application/json");
    });

    // ── POST /v1/audio/speech ──
    srv.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        // ── Parse JSON body ──
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error &) {
            res.status = 400;
            res.set_content(json_error("Invalid JSON body", "invalid_request_error", "", ""), "application/json");
            return;
        }

        // ── Validate required fields ──
        if (!body.contains("input") || !body["input"].is_string() || body["input"].get<std::string>().empty()) {
            res.status = 400;
            res.set_content(json_error("input is required", "invalid_request_error", "input", ""), "application/json");
            return;
        }

        if (!body.contains("voice") || !body["voice"].is_string()) {
            res.status = 400;
            res.set_content(json_error("voice is required", "invalid_request_error", "voice", ""), "application/json");
            return;
        }

        std::string input_text = body["input"].get<std::string>();
        std::string voice_name = body["voice"].get<std::string>();
        std::string model_req  = body.value("model", model_name_);

        // ── Validate model ──
        if (model_req != model_name_ && model_req != "tts-1" && model_req != "tts-1-hd") {
            // Accept generic OpenAI model names for compatibility
            // Just log a warning, don't reject
            printf("[server] Note: requested model '%s', using '%s'\n", model_req.c_str(), model_name_.c_str());
        }

        // ── Validate voice ──
        auto voice_it = voice_cache_.find(voice_name);
        if (voice_it == voice_cache_.end()) {
            res.status = 400;
            res.set_content(json_error(
                "Invalid voice: '" + voice_name + "'. Available: " +
                [this]() {
                    std::string names;
                    for (const auto & kv : voice_cache_) {
                        if (!names.empty()) names += ", ";
                        names += kv.first;
                    }
                    return names;
                }(),
                "invalid_request_error", "voice", "invalid_voice"
            ), "application/json");
            return;
        }

        // ── Parse optional fields ──
        std::string response_format = body.value("response_format", "wav");
        float speed = body.value("speed", 1.0f);

        if (speed != 1.0f) {
            // For MVP, log but don't error — speed not supported yet
            printf("[server] Note: speed=%.2f requested, but speed control is not supported. Using 1.0.\n", speed);
        }

        if (response_format != "wav" && response_format != "pcm" && response_format != "mp3") {
            res.status = 400;
            res.set_content(json_error(
                "Unsupported response_format: '" + response_format + "'. Supported: wav, pcm, mp3",
                "invalid_request_error", "response_format", "unsupported_format"
            ), "application/json");
            return;
        }

        if (response_format == "mp3" && !ffmpeg_available_) {
            res.status = 400;
            res.set_content(json_error(
                "MP3 format unavailable: ffmpeg not found on server PATH",
                "invalid_request_error", "response_format", "mp3_unavailable"
            ), "application/json");
            return;
        }

        // ── Generate audio ──
        printf("[server] Generating: voice='%s', text='%s'\n",
               voice_name.c_str(), input_text.c_str());

        std::vector<float> audio;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            auto t_start = std::chrono::high_resolution_clock::now();

            audio = pipeline_.generate_from_latent(input_text, voice_it->second, sampler_defaults_);

            auto t_end = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(t_end - t_start).count();
            printf("[server] Generated %.1f sec audio in %.1f seconds\n",
                   audio.size() / 44100.0f, elapsed);
        }

        if (audio.empty()) {
            res.status = 500;
            res.set_content(json_error("Audio generation failed", "server_error", "", ""), "application/json");
            return;
        }

        // ── Encode and send response ──
        if (response_format == "mp3") {
            auto mp3_bytes = audio_to_mp3_bytes(audio, 44100);
            if (mp3_bytes.empty()) {
                res.status = 500;
                res.set_content(json_error("MP3 encoding failed", "server_error", "", ""), "application/json");
                return;
            }
            res.set_content(
                std::string(reinterpret_cast<const char *>(mp3_bytes.data()), mp3_bytes.size()),
                "audio/mpeg"
            );
        } else if (response_format == "wav") {
            auto wav_bytes = audio_to_wav_bytes(audio, 44100);
            res.set_content(
                std::string(reinterpret_cast<const char *>(wav_bytes.data()), wav_bytes.size()),
                "audio/wav"
            );
        } else { // pcm
            auto pcm_bytes = audio_to_pcm_bytes(audio);
            res.set_content(
                std::string(reinterpret_cast<const char *>(pcm_bytes.data()), pcm_bytes.size()),
                "audio/l16"
            );
        }

        res.set_header("Content-Disposition", "attachment; filename=\"speech." + response_format + "\"");
    });

    // ── Start listening ──
    printf("\n[server] Starting HTTP server on %s:%d\n", config.host.c_str(), config.port);
    printf("[server] Endpoints:\n");
    printf("[server]   POST /v1/audio/speech  — OpenAI-compatible TTS\n");
    printf("[server]   GET  /v1/audio/models   — List models\n");
    printf("[server]   GET  /health            — Health check\n");
    printf("[server] Available voices: ");
    bool first = true;
    for (const auto & kv : voice_cache_) {
        printf("%s%s", first ? "" : ", ", kv.first.c_str());
        first = false;
    }
    printf("\n[server] Press Ctrl+C to stop.\n\n");

    // Install signal handler
    std::signal(SIGINT, signal_handler);

    bool ok = srv.listen(config.host.c_str(), config.port);
    g_http_server = nullptr;

    if (!ok) {
        fprintf(stderr, "[server] HTTP server stopped unexpectedly\n");
        return false;
    }

    printf("[server] Shutdown complete.\n");
    return true;
}

void EchoServer::stop() {
    if (g_http_server) {
        g_http_server->stop();
    }
}
