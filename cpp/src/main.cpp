// Echo-TTS C++ — CLI Entry Point
//
// Usage:
//   echo-tts --model echo-dit.gguf --speaker audio.wav --text "Hello world" [options]
//   echo-tts serve --model echo-dit.gguf --port 8080 --voice alloy=alloy.wav [...]

#include "echo_pipeline.h"
#include "echo_audio.h"
#include "echo_server.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ────────────────────────────────────────────────────────────────────
// Argument parsing helpers
// ────────────────────────────────────────────────────────────────────

static const char * get_arg(int argc, char ** argv, const char * flag, const char * default_val = nullptr) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], flag) == 0) {
            return argv[i + 1];
        }
    }
    return default_val;
}

static bool has_flag(int argc, char ** argv, const char * flag) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

static float get_float(int argc, char ** argv, const char * flag, float default_val) {
    const char * val = get_arg(argc, argv, flag);
    return val ? (float)atof(val) : default_val;
}

static int get_int(int argc, char ** argv, const char * flag, int default_val) {
    const char * val = get_arg(argc, argv, flag);
    return val ? atoi(val) : default_val;
}

// Parse comma-separated integers: "128,128,64" → {128, 128, 64}
static std::vector<int> parse_int_list(const char * str) {
    std::vector<int> result;
    if (!str) return result;
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        std::string token = s.substr(pos, comma - pos);
        if (!token.empty()) {
            result.push_back(atoi(token.c_str()));
        }
        pos = comma + 1;
    }
    return result;
}

// Collect all values for a repeated flag (e.g. --voice alloy=alloy.wav --voice echo=echo.wav)
static std::vector<const char *> get_all_args(int argc, char ** argv, const char * flag) {
    std::vector<const char *> result;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], flag) == 0) {
            result.push_back(argv[i + 1]);
        }
    }
    return result;
}

// Parse a single "name=path" pair from a --voice argument
static bool parse_voice_pair(const char * arg, std::string & name, std::string & path) {
    const char * eq = strchr(arg, '=');
    if (!eq || eq == arg || *(eq + 1) == '\0') return false;
    name.assign(arg, eq - arg);
    path.assign(eq + 1);
    return true;
}

static void print_usage(const char * prog) {
    printf("Echo-TTS C++ Inference Engine\n\n");
    printf("Usage:\n");
    printf("  %s [options]                                  (single generation)\n", prog);
    printf("  %s serve --model MODEL --voice NAME=PATH [...] (HTTP server mode)\n\n", prog);
    printf("CLI generation options:\n");
    printf("  --model PATH         GGUF model file\n");
    printf("  --speaker PATH       Speaker reference WAV\n");
    printf("  --text \"...\"          Text to synthesize\n\n");
    printf("  --dac-encoder PATH   DAC encoder ONNX file\n");
    printf("  --dac-decoder PATH   DAC decoder ONNX file\n");
    printf("  --output PATH        Output WAV path (default: output.wav)\n");
    printf("  --steps N            Euler steps (default: 40)\n");
    printf("  --cfg-text F         Text CFG scale (default: 3.0)\n");
    printf("  --cfg-speaker F      Speaker CFG scale (default: 8.0)\n");
    printf("  --cfg-min-t F        CFG min timestep (default: 0.5)\n");
    printf("  --cfg-max-t F        CFG max timestep (default: 1.0)\n");
    printf("  --seed N             RNG seed (default: 0)\n");
    printf("  --seq-length N       Sequence length (default: 640, max 640)\n");
    printf("  --truncation F       Noise truncation factor (default: 0, disabled)\n");
    printf("  --rescale-k F        Temporal rescaling k (default: 0, disabled)\n");
    printf("  --rescale-sigma F    Temporal rescaling sigma (default: 0, disabled)\n");
    printf("  --blockwise N,N,...  Blockwise mode with block sizes\n");
    printf("  --continuation PATH  Continuation audio for blockwise\n");
    printf("  --dump-intermediates DIR  Dump intermediate tensors for debugging\n");
    printf("\n");
    printf("Server options (with 'serve' subcommand):\n");
    printf("  --max-chunk-chars N   Max chars per text chunk (default: 400, 0=disable)\n");
    printf("  --port N              HTTP port (default: 8080)\n");
    printf("  --host IP             Listen address (default: 0.0.0.0)\n");
    printf("  --voice NAME=PATH     Register a voice (repeatable)\n");
    printf("  --help               Show this help\n");
}

// ────────────────────────────────────────────────────────────────────
// Serve subcommand
// ────────────────────────────────────────────────────────────────────

static int cmd_serve(int argc, char ** argv) {
    const char * model_path       = get_arg(argc, argv, "--model");
    const char * dac_encoder_path = get_arg(argc, argv, "--dac-encoder", "");
    const char * dac_decoder_path = get_arg(argc, argv, "--dac-decoder", "");
    const char * host             = get_arg(argc, argv, "--host", "0.0.0.0");
    int port                      = get_int(argc, argv, "--port", 8080);

    if (!model_path) {
        fprintf(stderr, "ERROR: --model is required for serve mode\n");
        return 1;
    }
    if (!dac_encoder_path || !dac_encoder_path[0]) {
        fprintf(stderr, "ERROR: --dac-encoder is required for serve mode\n");
        return 1;
    }
    if (!dac_decoder_path || !dac_decoder_path[0]) {
        fprintf(stderr, "ERROR: --dac-decoder is required for serve mode\n");
        return 1;
    }

    // Parse --voice name=path pairs
    std::vector<const char *> voice_args = get_all_args(argc, argv, "--voice");
    if (voice_args.empty()) {
        fprintf(stderr, "ERROR: At least one --voice name=path is required for serve mode\n");
        return 1;
    }

    std::unordered_map<std::string, std::string> voices;
    for (const char * va : voice_args) {
        std::string name, path;
        if (!parse_voice_pair(va, name, path)) {
            fprintf(stderr, "ERROR: Invalid --voice format '%s'. Use: --voice name=path\n", va);
            return 1;
        }
        voices[name] = path;
    }

    // Build server config
    EchoServerConfig server_config;
    server_config.host             = host;
    server_config.port             = port;
    server_config.model_path       = model_path;
    server_config.dac_encoder_path = dac_encoder_path;
    server_config.dac_decoder_path = dac_decoder_path;
    server_config.voices           = voices;

    // Sampler defaults
    server_config.sampler_defaults.num_steps         = get_int(argc, argv, "--steps", 40);
    server_config.sampler_defaults.cfg_scale_text    = get_float(argc, argv, "--cfg-text", 3.0f);
    server_config.sampler_defaults.cfg_scale_speaker = get_float(argc, argv, "--cfg-speaker", 8.0f);
    server_config.sampler_defaults.cfg_min_t         = get_float(argc, argv, "--cfg-min-t", 0.5f);
    server_config.sampler_defaults.cfg_max_t         = get_float(argc, argv, "--cfg-max-t", 1.0f);
    server_config.sampler_defaults.rng_seed          = (uint64_t)get_int(argc, argv, "--seed", 0);
    server_config.sampler_defaults.sequence_length   = get_int(argc, argv, "--seq-length", 640);
    server_config.max_chunk_chars                     = get_int(argc, argv, "--max-chunk-chars", 400);

    // Start server (blocks until shutdown)
    EchoServer server;
    if (!server.start(server_config)) {
        fprintf(stderr, "ERROR: Server failed to start\n");
        return 1;
    }

    return 0;
}

// ────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    // Route to serve subcommand
    if (strcmp(argv[1], "serve") == 0) {
        return cmd_serve(argc, argv);
    }

    // ── CLI Generation Mode ──

    // Parse arguments
    const char * model_path       = get_arg(argc, argv, "--model");
    const char * speaker_path     = get_arg(argc, argv, "--speaker");
    const char * text             = get_arg(argc, argv, "--text");
    const char * dac_encoder_path = get_arg(argc, argv, "--dac-encoder", "");
    const char * dac_decoder_path = get_arg(argc, argv, "--dac-decoder", "");
    const char * output_path      = get_arg(argc, argv, "--output", "output.wav");
    const char * blockwise_str    = get_arg(argc, argv, "--blockwise");
    const char * continuation_path = get_arg(argc, argv, "--continuation", "");
    const char * dump_dir          = get_arg(argc, argv, "--dump-intermediates");

    // Validate required args
    if (!model_path) {
        fprintf(stderr, "ERROR: --model is required\n");
        return 1;
    }
    if (!text) {
        fprintf(stderr, "ERROR: --text is required\n");
        return 1;
    }
    if (!speaker_path) {
        fprintf(stderr, "ERROR: --speaker is required\n");
        return 1;
    }

    // Build sampler config
    EchoSamplerConfig sampler;
    sampler.num_steps         = get_int(argc, argv, "--steps", 40);
    sampler.cfg_scale_text    = get_float(argc, argv, "--cfg-text", 3.0f);
    sampler.cfg_scale_speaker = get_float(argc, argv, "--cfg-speaker", 8.0f);
    sampler.cfg_min_t         = get_float(argc, argv, "--cfg-min-t", 0.5f);
    sampler.cfg_max_t         = get_float(argc, argv, "--cfg-max-t", 1.0f);
    sampler.rng_seed          = (uint64_t)get_int(argc, argv, "--seed", 0);
    sampler.sequence_length   = get_int(argc, argv, "--seq-length", 640);
    sampler.truncation_factor = get_float(argc, argv, "--truncation", 0.0f);
    sampler.rescale_k         = get_float(argc, argv, "--rescale-k", 0.0f);
    sampler.rescale_sigma     = get_float(argc, argv, "--rescale-sigma", 0.0f);

    // Print config
    printf("═══════════════════════════════════════════\n");
    printf("  Echo-TTS C++ Inference Engine\n");
    printf("═══════════════════════════════════════════\n");
    printf("  Model:    %s\n", model_path);
    printf("  Speaker:  %s\n", speaker_path);
    printf("  Text:     %s\n", text);
    printf("  Output:   %s\n", output_path);
    printf("  Steps:    %d\n", sampler.num_steps);
    printf("  CFG:      text=%.1f, speaker=%.1f (t=[%.2f, %.2f])\n",
           sampler.cfg_scale_text, sampler.cfg_scale_speaker,
           sampler.cfg_min_t, sampler.cfg_max_t);
    printf("  Seed:     %llu\n", (unsigned long long)sampler.rng_seed);
    printf("  Seq len:  %d\n", sampler.sequence_length);
    printf("═══════════════════════════════════════════\n\n");

    // Build pipeline config
    EchoPipelineConfig pipeline_config;
    pipeline_config.model_path = model_path;
    pipeline_config.dac_encoder_path = dac_encoder_path;
    pipeline_config.dac_decoder_path = dac_decoder_path;

    // Load pipeline
    EchoPipeline pipeline;
    if (!pipeline.load(pipeline_config)) {
        fprintf(stderr, "ERROR: Failed to load pipeline\n");
        return 1;
    }

    // Diagnostic dump mode
    if (dump_dir) {
        printf("═══ Diagnostic Dump Mode ═══\n");
        printf("  Output dir: %s\n", dump_dir);
        if (pipeline.diagnostic_dump(dump_dir, text, speaker_path, sampler)) {
            printf("\n✓ Diagnostic dumps saved to %s/\n", dump_dir);
            return 0;
        } else {
            fprintf(stderr, "ERROR: Diagnostic dump failed\n");
            return 1;
        }
    }

    // Generate
    std::vector<float> audio;
    auto t_start = std::chrono::high_resolution_clock::now();

    if (blockwise_str) {
        // Blockwise mode
        std::vector<int> block_sizes = parse_int_list(blockwise_str);
        if (block_sizes.empty()) {
            fprintf(stderr, "ERROR: Invalid --blockwise format. Use: 128,128,64\n");
            return 1;
        }

        printf("Blockwise mode: %zu blocks [", block_sizes.size());
        for (size_t i = 0; i < block_sizes.size(); i++) {
            printf("%d%s", block_sizes[i], (i + 1 < block_sizes.size()) ? "," : "");
        }
        printf("]\n\n");

        EchoBlockwiseConfig bw_config;
        bw_config.base = sampler;
        bw_config.block_sizes = block_sizes;

        audio = pipeline.generate_blockwise(text, speaker_path, bw_config, continuation_path);
    } else {
        // Standard mode
        audio = pipeline.generate(text, speaker_path, sampler);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t_end - t_start).count();

    // Save output
    if (!audio.empty()) {
        if (save_wav(output_path, audio.data(), (int)audio.size(), 44100)) {
            printf("\n✓ Saved output to: %s (%.1f sec, %.1f sec generation time)\n",
                   output_path, audio.size() / 44100.0f, elapsed);
        } else {
            fprintf(stderr, "ERROR: Failed to save WAV to: %s\n", output_path);
            return 1;
        }
    } else {
        fprintf(stderr, "ERROR: No audio generated\n");
        return 1;
    }

    return 0;
}
