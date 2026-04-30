# Echo-TTS C++

A high-performance C++ inference engine for [Echo-TTS](https://github.com/jordandarefsky/echo-tts), a multi-speaker text-to-speech model with speaker reference conditioning. Uses [GGML](https://github.com/ggml-org/ggml) for the diffusion transformer and ONNX Runtime for the S1-DAC autoencoder — runs entirely on GPU via CUDA.

**Original model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) by [Jordan Darefsky](https://jordandarefsky.com/blog/2025/echo/).

## Requirements

- NVIDIA GPU with CUDA ≥12.x (tested on RTX 4080)
- CMake ≥3.18
- MSVC 2022 (Windows) or GCC 11+ (Linux)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (GPU package, ≥1.19)

GGML is fetched automatically by CMake via `FetchContent` — no manual setup needed.

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
cmake .. -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOT=C:/path/to/onnxruntime-win-x64-gpu-1.19.0
cmake --build . --config Release --parallel

# Linux
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.19.0
cmake --build . --config Release --parallel
```

ONNX Runtime is optional — if omitted, the DAC autoencoder is skipped (useful for diagnostic dumps only).

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
