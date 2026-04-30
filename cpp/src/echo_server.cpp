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
#include <memory>

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
// Text splitting for TTS (sentence-aware chunking)
// ────────────────────────────────────────────────────────────────────

std::vector<std::string> EchoServer::split_text_for_tts(const std::string & text, int max_chunk_chars) {
    if (max_chunk_chars <= 0 || text.empty()) {
        return { text };
    }

    std::vector<std::string> raw_sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        current += c;

        bool is_delim = (c == '.' || c == '!' || c == '?' || c == ';' || c == '\n');
        bool is_quote_end = (i + 1 < text.size() && (c == '"' || c == '\'')) &&
                            text[i + 1] == ' ';
        bool is_ellipsis = (c == '.' && i + 2 < text.size() &&
                            text[i + 1] == '.' && text[i + 2] == '.');

        if (is_delim && !is_ellipsis) {
            if (!current.empty()) {
                raw_sentences.push_back(current);
                current.clear();
            }
        } else if (is_quote_end) {
            if (!current.empty()) {
                raw_sentences.push_back(current);
                current.clear();
            }
        } else if (is_ellipsis) {
            current += "..";
            i += 2;
        }
    }
    if (!current.empty()) {
        raw_sentences.push_back(current);
    }

    if (raw_sentences.empty()) {
        return { text };
    }

    // Merge short sentences into larger chunks, respecting max_chunk_chars
    std::vector<std::string> merged;
    std::string buffer;

    for (const auto & sentence : raw_sentences) {
        std::string trimmed = sentence;
        while (!trimmed.empty() && (trimmed[0] == ' ' || trimmed[0] == '\n')) {
            trimmed.erase(0, 1);
        }

        if ((int)trimmed.size() >= max_chunk_chars) {
            if (!buffer.empty()) {
                merged.push_back(buffer);
                buffer.clear();
            }
            // Long sentence: try splitting on commas
            std::string sub_buffer;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                sub_buffer += c;
                bool break_point = (c == ',') || (c == ')') || (c == '—') || (c == '–');
                if (break_point && (int)sub_buffer.size() >= max_chunk_chars / 4 && i + 1 < trimmed.size()) {
                    merged.push_back(sub_buffer);
                    sub_buffer.clear();
                }
            }
            if (!sub_buffer.empty()) {
                if (!buffer.empty() && (int)(buffer.size() + sub_buffer.size()) <= max_chunk_chars) {
                    buffer += sub_buffer;
                } else {
                    if (!buffer.empty()) {
                        merged.push_back(buffer);
                        buffer.clear();
                    }
                    // Hard split if still too long
                    if ((int)sub_buffer.size() > max_chunk_chars) {
                        for (size_t pos = 0; pos < sub_buffer.size(); pos += max_chunk_chars) {
                            size_t len = std::min((size_t)max_chunk_chars, sub_buffer.size() - pos);
                            merged.push_back(sub_buffer.substr(pos, len));
                        }
                    } else {
                        buffer = sub_buffer;
                    }
                }
            }
        } else {
            if (!buffer.empty() && (int)(buffer.size() + trimmed.size()) > max_chunk_chars) {
                merged.push_back(buffer);
                buffer = trimmed;
            } else {
                if (!buffer.empty()) buffer += " ";
                buffer += trimmed;
            }
        }
    }
    if (!buffer.empty()) {
        merged.push_back(buffer);
    }

    if (merged.empty()) {
        return { text };
    }

    return merged;
}

// ────────────────────────────────────────────────────────────────────
// Multi-chunk audio generation
// ────────────────────────────────────────────────────────────────────

std::vector<float> EchoServer::generate_chunked_audio(
    const std::string & text,
    const SpeakerLatentData & speaker,
    const EchoSamplerConfig & sampler_config,
    bool log_progress
) {
    int max_chars = max_chunk_chars_;

    // Disable chunking if max_chunk_chars is 0 or if text is short enough
    if (max_chars <= 0 || (int)text.size() <= max_chars) {
        return pipeline_.generate_from_latent(text, speaker, sampler_config);
    }

    std::vector<std::string> chunks = split_text_for_tts(text, max_chars);

    if (chunks.size() <= 1) {
        return pipeline_.generate_from_latent(text, speaker, sampler_config);
    }

    if (log_progress) {
        printf("[server] Text split into %zu chunks:\n", chunks.size());
        for (size_t i = 0; i < chunks.size(); i++) {
            printf("[server]   [%zu/%zu] %zu chars: \"%s\"\n",
                   i + 1, chunks.size(), chunks[i].size(), chunks[i].c_str());
        }
    }

    // Pre-compute speaker KV cache (shared across all chunks)
    EchoSamplerConfig chunk_config = sampler_config;
    if (log_progress) {
        printf("[server] Pre-computing speaker KV cache...\n");
    }
    EchoKVCache kv_speaker = pipeline_.compute_speaker_kv(speaker);

    // Generate each chunk
    std::vector<float> all_audio;
    const float SILENCE_DURATION = 0.1f;  // 100ms silence between chunks
    const int SILENCE_SAMPLES = (int)(SILENCE_DURATION * 44100);
    std::vector<float> silence(SILENCE_SAMPLES, 0.0f);

    for (size_t i = 0; i < chunks.size(); i++) {
        if (log_progress) {
            printf("[server] Generating chunk %zu/%zu (%zu chars)...\n",
                   i + 1, chunks.size(), chunks[i].size());
        }

        std::vector<float> chunk_audio = pipeline_.generate_from_latent_with_speaker_kv(
            chunks[i], speaker, chunk_config, kv_speaker
        );

        if (chunk_audio.empty()) {
            fprintf(stderr, "[server] WARNING: Chunk %zu generated no audio\n", i + 1);
            continue;
        }

        // Append chunk audio
        all_audio.insert(all_audio.end(), chunk_audio.begin(), chunk_audio.end());

        // Insert silence between chunks (not after last)
        if (i + 1 < chunks.size()) {
            all_audio.insert(all_audio.end(), silence.begin(), silence.end());
        }
    }

    if (log_progress) {
        printf("[server] Chunked generation complete: %zu chunks → %.1f sec total\n",
               chunks.size(), all_audio.size() / 44100.0f);
    }

    return all_audio;
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
    max_chunk_chars_ = config.max_chunk_chars;

    // ── Load pipeline ──
    EchoPipelineConfig pipeline_config;
    pipeline_config.model_path       = config.model_path;
    pipeline_config.dac_encoder_path = config.dac_encoder_path;
    pipeline_config.dac_decoder_path = config.dac_decoder_path;

    if (!pipeline_.load(pipeline_config)) {
        fprintf(stderr, "[server] ERROR: Failed to load pipeline\n");
        return false;
    }

    if (config.log_vram) {
        EchoModel::log_vram("server-after-model-load");
        log_vram_ = true;
    }

    // ── Pre-encode voices ──
    if (!pre_encode_voices(config.voices)) {
        fprintf(stderr, "[server] ERROR: Failed to pre-encode voices\n");
        return false;
    }

    if (log_vram_) {
        EchoModel::log_vram("server-after-voices-encoded");
    }

    // Release ORT encoder session — never needed again after voice pre-encoding
    pipeline_.release_dac_encoder();

    if (log_vram_) {
        EchoModel::log_vram("server-after-encoder-released");
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

        if (log_vram_) {
            EchoModel::log_vram("server-before-generation");
        }

        std::vector<float> audio;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            auto t_start = std::chrono::high_resolution_clock::now();

            audio = generate_chunked_audio(input_text, voice_it->second, sampler_defaults_, true);

            auto t_end = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(t_end - t_start).count();
            printf("[server] Generated %.1f sec audio in %.1f seconds\n",
                   audio.size() / 44100.0f, elapsed);
        }

        if (log_vram_) {
            EchoModel::log_vram("server-after-generation");
        }

        pipeline_.release_scheduler_memory();

        if (log_vram_) {
            EchoModel::log_vram("server-after-scheduler-reset");
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

    // ── POST /v1/audio/speech/stream (SSE pseudo-streaming) ──
    srv.Post("/v1/audio/speech/stream", [this](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        // Parse JSON body
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error &) {
            res.status = 400;
            res.set_content(json_error("Invalid JSON body", "invalid_request_error", "", ""), "application/json");
            return;
        }

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

        auto voice_it = voice_cache_.find(voice_name);
        if (voice_it == voice_cache_.end()) {
            res.status = 400;
            res.set_content(json_error("Invalid voice: '" + voice_name + "'", "invalid_request_error", "voice", "invalid_voice"), "application/json");
            return;
        }

        std::string response_format = body.value("response_format", "wav");

        // Split text into chunks
        int max_chars = max_chunk_chars_;
        if (max_chars <= 0) max_chars = 400;
        std::vector<std::string> chunks = split_text_for_tts(input_text, max_chars);
        if (chunks.empty()) chunks.push_back(input_text);

        printf("[server/stream] Generating: voice='%s', %zu chunk(s), text='%s'\n",
               voice_name.c_str(), chunks.size(), input_text.c_str());

        // Use chunked content provider for SSE streaming
        struct StreamState {
            EchoServer * server;
            SpeakerLatentData speaker_data;
            EchoSamplerConfig config;
            std::vector<std::string> chunks;
            std::string format;
            size_t current_idx = 0;
            bool speaker_kv_computed = false;
            bool done = false;
            EchoKVCache kv_speaker;
        };

        auto state = std::make_shared<StreamState>();
        state->server = this;
        state->speaker_data = voice_it->second;
        state->config = sampler_defaults_;
        state->chunks = chunks;
        state->format = response_format;

        res.set_chunked_content_provider("text/event-stream",
            [state](size_t /*offset*/, httplib::DataSink & sink) -> bool {
                while (state->current_idx < state->chunks.size()) {
                    // Serialize GPU access within the callback
                    std::lock_guard<std::mutex> lock(state->server->gpu_mutex_);

                    if (!state->speaker_kv_computed) {
                        printf("[server/stream] Pre-computing speaker KV cache...\n");
                        state->kv_speaker = state->server->pipeline_.compute_speaker_kv(state->speaker_data);
                        state->speaker_kv_computed = true;
                    }

                    const auto & chunk_text = state->chunks[state->current_idx];
                    printf("[server/stream] Generating chunk %zu/%zu (%zu chars)...\n",
                           state->current_idx + 1, state->chunks.size(), chunk_text.size());

                    std::vector<float> chunk_audio =
                        state->server->pipeline_.generate_from_latent_with_speaker_kv(
                            chunk_text, state->speaker_data, state->config, state->kv_speaker
                        );

                    // Encode chunk as WAV bytes then base64
                    std::vector<uint8_t> audio_bytes;
                    if (state->format == "pcm") {
                        audio_bytes = audio_to_pcm_bytes(chunk_audio);
                    } else {
                        audio_bytes = audio_to_wav_bytes(chunk_audio, 44100);
                    }

                    // Manual base64 encoding
                    static const char b64_table[] =
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    std::string audio_b64;
                    audio_b64.reserve((audio_bytes.size() + 2) / 3 * 4);
                    size_t i = 0;
                    while (i + 2 < audio_bytes.size()) {
                        uint32_t triple = ((uint32_t)audio_bytes[i] << 16)
                                        | ((uint32_t)audio_bytes[i + 1] << 8)
                                        | (uint32_t)audio_bytes[i + 2];
                        audio_b64 += b64_table[(triple >> 18) & 0x3F];
                        audio_b64 += b64_table[(triple >> 12) & 0x3F];
                        audio_b64 += b64_table[(triple >> 6) & 0x3F];
                        audio_b64 += b64_table[triple & 0x3F];
                        i += 3;
                    }
                    if (i < audio_bytes.size()) {
                        uint32_t triple = (uint32_t)audio_bytes[i] << 16;
                        if (i + 1 < audio_bytes.size()) triple |= (uint32_t)audio_bytes[i + 1] << 8;
                        audio_b64 += b64_table[(triple >> 18) & 0x3F];
                        audio_b64 += b64_table[(triple >> 12) & 0x3F];
                        audio_b64 += (i + 1 < audio_bytes.size())
                            ? b64_table[(triple >> 6) & 0x3F] : '=';
                        audio_b64 += '=';
                    }

                    json event;
                    event["chunk"] = state->current_idx + 1;
                    event["total"] = state->chunks.size();
                    event["text"]  = chunk_text;
                    event["audio_b64"] = audio_b64;
                    event["format"] = state->format;

                    std::string sse_data = "data: " + event.dump() + "\n\n";
                    state->current_idx++;

                    bool ok = sink.write(sse_data.data(), sse_data.size());
                    if (!ok) return false;
                }

                // All chunks done — send completion event
                if (!state->done) {
                    state->done = true;
                    std::string done_event = "data: {\"done\":true}\n\n";
                    sink.write(done_event.data(), done_event.size());
                    sink.done();
                }
                return true;
            });
    });

    // ── Start listening ──
    printf("\n[server] Starting HTTP server on %s:%d\n", config.host.c_str(), config.port);
    printf("[server] Endpoints:\n");
    printf("[server]   POST /v1/audio/speech         — OpenAI-compatible TTS (with auto-chunking)\n");
    printf("[server]   POST /v1/audio/speech/stream  — SSE pseudo-streaming\n");
    printf("[server]   GET  /v1/audio/models          — List models\n");
    printf("[server]   GET  /health                   — Health check\n");
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
