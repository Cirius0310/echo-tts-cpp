# Echo-TTS C++

A high-performance C++ inference engine for [Echo-TTS](https://github.com/jordandarefsky/echo-tts), a multi-speaker text-to-speech model with speaker reference conditioning. Uses [GGML](https://github.com/ggml-org/ggml) for the diffusion transformer and ONNX Runtime for the S1-DAC autoencoder — runs entirely on GPU via CUDA.

**Original model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) by [Jordan Darefsky](https://jordandarefsky.com/blog/2025/echo/).

## Requirements

- NVIDIA GPU with CUDA ≥12.x (tested on RTX 4080)
- CMake ≥3.18
- MSVC 2022 (Windows) or GCC 11+ (Linux)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (GPU package, tested with 1.25.0)
- [cuDNN 9.x](https://developer.nvidia.com/cudnn) — required at runtime by ONNX Runtime CUDA provider
- [FFmpeg](https://ffmpeg.org/download.html) — optional, required for MP3 output in server mode

GGML, cpp-httplib, and nlohmann/json are fetched automatically by CMake via `FetchContent` — no manual setup needed.

## Model Files

The GGUF model weights and ONNX autoencoder files are too large for git.

### Pre-converted model files

Download GGUF and ONNX DAC models directly from HuggingFace:

```bash
huggingface-cli download tmdarkbr/echo-tts-gguf echo-dit.gguf --local-dir .
huggingface-cli download tmdarkbr/echo-tts-gguf onnx/ --local-dir onnx_models
```

### Generate from scratch

If you prefer to convert the weights yourself, or to export the ONNX autoencoder:

```bash
# Clone the original Python repo (required)
git clone https://github.com/jordandarefsky/echo-tts.git ../echo-tts
pip install gguf safetensors huggingface-hub torch onnx onnxruntime einops
```

**GGUF model weights:**
```bash
python scripts/export_gguf.py --output echo-dit.gguf --dtype f16
```

**ONNX autoencoder models:**
```bash
python scripts/export_dac_onnx.py --echo-tts-path ../echo-tts --output-dir onnx_models
```

### Speaker reference audio

Any 44100Hz mono WAV file (up to 5 minutes). The included `kore_gemini.wav` and `ana_eleven.wav` are examples.

## Build

```bash
cd cpp
mkdir build && cd build

# Windows (MSVC)
cmake .. -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOT=C:/path/to/onnxruntime-win-x64-gpu-1.25.0
cmake --build . --config Release --parallel

# Linux
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.25.0
cmake --build . --config Release --parallel
```

ONNX Runtime is optional — if omitted, the DAC autoencoder is skipped (useful for diagnostic dumps only).

**Runtime DLLs:** The build copies ONNX Runtime and GGML DLLs next to the executable automatically. You still need these libraries accessible (in `PATH` or next to the exe):

- **[cuDNN 9.x](https://developer.nvidia.com/cudnn)** — `cudnn64_9.dll` plus engine DLLs
- **CUDA 12.x runtime** — `cublas64_12.dll`, `cublasLt64_12.dll`, etc. (ONNX Runtime 1.25.0 is built against CUDA 12)

If your system has CUDA 12.x already (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`), add both to `PATH`:

```powershell
[Environment]::SetEnvironmentVariable("PATH",
    $env:PATH + ";C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64" +
    ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    "User")
```

Alternatively, copy the needed `.dll` files directly next to `echo-tts.exe`.

## Usage

```bash
# Basic generation
echo-tts \
  --model echo-dit.gguf \
  --speaker kore_gemini.wav \
  --dac-encoder onnx_models/dac_encoder.onnx \
  --dac-decoder onnx_models/dac_decoder.onnx \
  --text "[S1] Hello, this is a test of the Echo TTS model." \
  --output output.wav
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to GGUF model file |
| `--speaker` | *(required)* | Speaker reference WAV (44100Hz mono) |
| `--text` | *(required)* | Text to synthesize |
| `--dac-encoder` | — | DAC encoder ONNX model |
| `--dac-decoder` | — | DAC decoder ONNX model |
| `--output` | `output.wav` | Output WAV path |
| `--steps` | `40` | Euler sampling steps |
| `--cfg-text` | `3.0` | Text CFG scale |
| `--cfg-speaker` | `8.0` | Speaker CFG scale |
| `--cfg-min-t` | `0.5` | CFG min timestep |
| `--cfg-max-t` | `1.0` | CFG max timestep |
| `--seed` | `0` | RNG seed (Mersenne Twister) |
| `--seq-length` | `640` | Sequence length (max 640, ~30 sec) |
| `--truncation` | `0` | Noise truncation factor |
| `--blockwise` | — | Comma-separated block sizes (e.g. `128,128,64`) |
| `--continuation` | — | Continuation WAV for blockwise mode |
| `--dump-intermediates` | — | Dump intermediate tensors for debugging |

## Server Mode

`echo-tts serve` starts an HTTP server with an OpenAI-compatible TTS API. Voices are pre-encoded on startup for low-latency generation.

### Quick Start

```bash
# Windows
echo-tts serve ^
  --model echo-dit.gguf ^
  --dac-encoder onnx_models/dac_encoder.onnx ^
  --dac-decoder onnx_models/dac_decoder.onnx ^
  --voice alloy=kore_gemini.wav ^
  --voice echo=ana_eleven.wav ^
  --port 8080

# Linux
echo-tts serve \
  --model echo-dit.gguf \
  --dac-encoder onnx_models/dac_encoder.onnx \
  --dac-decoder onnx_models/dac_decoder.onnx \
  --voice alloy=kore_gemini.wav \
  --voice echo=ana_eleven.wav \
  --port 8080
```

### Server CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | GGUF model file |
| `--dac-encoder` | *(required)* | DAC encoder ONNX model |
| `--dac-decoder` | *(required)* | DAC decoder ONNX model |
| `--voice` | *(required)* | `name=path` pair (repeatable, e.g. `--voice alloy=speaker.wav`) |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `8080` | Listen port |
| `--steps` | `40` | Euler sampling steps |
| `--cfg-text` | `3.0` | Text CFG scale |
| `--cfg-speaker` | `8.0` | Speaker CFG scale |
| `--seed` | `0` | RNG seed |
| `--seq-length` | `640` | Sequence length |

### API Endpoints

**`POST /v1/audio/speech`** — OpenAI-compatible TTS

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model name (e.g. `"echo-tts"`, `"tts-1"`, `"tts-1-hd"`) |
| `input` | string | *(required)* | Text to synthesize (max ~4096 characters) |
| `voice` | string | *(required)* | Voice name configured at startup |
| `response_format` | string | `"wav"` | Output format: `wav`, `pcm`, or `mp3` (requires ffmpeg) |
| `speed` | number | `1.0` | Playback speed (not currently supported) |

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"echo-tts","input":"Hello world.","voice":"alloy","response_format":"mp3"}' \
  -o output.mp3
```

Error responses follow the OpenAI format:
```json
{"error":{"message":"Invalid voice: 'foo'","type":"invalid_request_error","param":"voice","code":"invalid_voice"}}
```

**`GET /v1/audio/models`** — List available models

```bash
curl http://localhost:8080/v1/audio/models
```

**`GET /health`** — Health check with voice list

```bash
curl http://localhost:8080/health
```

### SillyTavern Configuration

Set the TTS provider to **OpenAI** and configure:

- **Provider Endpoint:** `http://127.0.0.1:8080/v1/audio/speech`
- **Model:** `echo-tts`
- **Voice:** Pick any voice name from your `--voice` flags

## Examples

Pre-generated audio samples showcasing different voices and text styles — browse them in the [`examples/`](examples/) directory:

| File | Voice | Duration | Style |
|------|-------|----------|-------|
| [`01_echo_welcome.wav`](examples/01_echo_welcome.wav) | echo | 13.6s | Product intro narration |
| [`02_herta_dialogue.wav`](examples/02_herta_dialogue.wav) | herta | 12.8s | Casual dialogue |
| [`03_echo_technical.wav`](examples/03_echo_technical.wav) | echo | 20.0s | Technical explanation |
| [`04_herta_playful.wav`](examples/04_herta_playful.wav) | herta | 13.7s | Playful fourth-wall break |
| [`05_echo_architecture.wav`](examples/05_echo_architecture.wav) | echo | 29.7s | Architecture deep dive (max capacity) |
| [`06_kore_wisdom.wav`](examples/06_kore_wisdom.wav) | kore | 15.4s | Formal wisdom-keeper persona |

All generated on an RTX 4080 SUPER — see the [examples README](examples/README.md) for full generation stats and voice descriptions.

## Architecture

| Component | Backend | Details |
|-----------|---------|---------|
| EchoDiT transformer | GGML (CUDA) | 24-layer DiT decoder, 2×14-layer encoders, joint attention + Low-Rank AdaLN |
| S1-DAC autoencoder | ONNX Runtime (CUDA) | 1024-dim latent, 2048× downsample |
| PCA compression | CPU (custom) | 1024→80 dim, latent_scale=0.0556 |
| Tokenizer | CPU (custom) | Byte-level BPE, 256 vocab (WhisperD format) |
| Sampler | CPU/GPU | Euler ODE, 3-pass independent CFG, blockwise support |

## Differences from Python

- **CFG batching:** 3 separate forward passes instead of batched (uses less VRAM, ~3× slower per step)
- **RNG:** Mersenne Twister (C++) vs CUDA Philox (Python) — different initial noise, output is not bit-identical
- **Audio resampling:** Linear interpolation vs sinc/Kaiser — speaker encoding has slight quality differences
- **GGML tensor layout:** Column-major weight storage (transposed from PyTorch row-major)

## Responsible Use

Don't use this to impersonate real people without consent or generate deceptive audio. You are responsible for complying with local laws regarding biometric data and voice cloning.

## License

Code is MIT-licensed (see [LICENSE](LICENSE)), matching the original Python codebase.

The `dr_wav.h` header in `cpp/extern/` is public domain / MIT.

Regardless of code license, audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the Fish Speech S1-DAC autoencoder. The Echo-TTS model weights are released under CC-BY-NC-SA-4.0.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```

## Acknowledgments

This project builds on:

- [Echo-TTS](https://github.com/jordandarefsky/echo-tts) by Jordan Darefsky
- [GGML](https://github.com/ggml-org/ggml) by ggerganov and contributors
- [Fish Speech S1-DAC](https://github.com/fishaudio/fish-speech) autoencoder
- [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) text format
