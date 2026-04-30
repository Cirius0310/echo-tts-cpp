# Echo-TTS C++ Engine — Debug Findings

## Overview

Comparing C++ inference engine output against Python reference using identical inputs (`"Hello world"`, `kore_gemini.wav`, seed=0, seq_len=640, t=0.5).

The C++ engine uses GGML (CUDA backend) for the EchoDiT transformer and ONNX Runtime for the S1-DAC autoencoder.

---

## Bugs Found & Fixed

### 0. Diagnostic x_t / model_fwd_out Layout — FIXED ✅

**Problem:** The diagnostic dump converted GGML `[latent_dim, seq]` memory using
`d * seq + s`. For a contiguous GGML tensor with `ne[0]=latent_dim`,
the memory offset is `s * latent_dim + d`, which is already Python
`(batch, seq, latent_dim)` row-major order for batch 1.

**Fix Applied:** `cpp/src/echo_pipeline.cpp` now dumps `x_t` and
`model_fwd_out` using `s * latent_dim + d`.

**Impact:** This does not explain the 1.84x model scale gap; after correcting the
dump layout, Python with C++ inputs still correlates with C++ at ~0.978 and C++
is still ~1.84x larger. It does make future same-input comparisons valid.

### 0b. GGML CUDA cond_module.4 Matrix-Vector Diagnostic — FIXED ✅

**Problem:** Instrumenting the timestep conditioning MLP showed:

| Tensor | Python std | C++ std |
|--------|------------|---------|
| cond_module.0 output | 0.3887 | 0.3886 |
| cond_module.2 output | 0.0211 | 0.0211 |
| cond_module.4 / cond_embed | 0.0177 | 2.85–3.02 |

The first two conditioning projections matched exactly. The final projection
(`cond_module.4.weight`, shape 6144 x 2048, stored F16 in GGUF) hit a bad GGML
CUDA matrix-vector path and produced a wildly over-scaled diagnostic tensor.

**Fix Applied:** `cpp/src/echo_model.cpp` now caches the three conditioning MLP
weights as CPU float vectors and computes the timestep conditioning on CPU, then
feeds `cond_embed` as an F32 graph input. With `ECHO_DEBUG_MODEL_STATS=1`,
C++ now reports `cond_embed std=0.017710`, matching Python.

**Impact:** Surprisingly, this was not the source of the final 1.84x output gap;
the model output remained ~2.97 std after fixing conditioning. Keep the fix
anyway because the previous conditioning graph was numerically wrong.

### 1. Token Padding — FIXED ✅

**Problem:** C++ tokenizer (`cpp/src/echo_tokenizer.cpp`) always padded token IDs to `max_length=768`. For "Hello world" (17 tokens), this fed 751 extra zero-padding tokens through the text encoder's **bidirectional** attention, corrupting the valid token representations. Python uses `pad_to_max=False` by default.

**Fix Applied:** Changed default `max_length` from 768 to 0 (no padding). When `max_length=0`, the tokenizer returns exactly the actual tokens with no padding. Files changed:
- `cpp/src/echo_tokenizer.h` — changed default param from 768 to 0
- `cpp/src/echo_tokenizer.cpp` — added branching logic for padded vs unpadded mode

**Verification:** After fix, token_ids stage passes with 0 errors, shapes match `(1, 17)`.

### 2. z_q Diagnostic Dump Layout — FIXED ✅

**Problem:** The diagnostic dump code in `cpp/src/echo_pipeline.cpp` (lines ~418-424) incorrectly transposed the DAC encoder's z_q output. It iterated `for t, for d` pushing `z_q[d*T+t]`, producing data in `(T, 1024)` order but labeling it as shape `(1, 1024, T)`.

The ONNX Runtime already returns z_q in standard row-major `(1, 1024, T)` order — no rearrangement needed.

**Fix Applied:** Replaced the loop with `all_zq_flat.insert(all_zq_flat.end(), z_q.begin(), z_q.end())`.

**Note:** This was a cosmetic/diagnostic-only bug. The actual PCA encoding pipeline uses the raw ONNX output correctly, so runtime inference was unaffected.

**Verification:** speaker_zq correlation improved from 0.007 (broken) to 0.90 (correct, remaining gap is from audio resampling difference).

---

### 3. Flash Attention Output Permute — FIXED ✅

**Problem:** After `ggml_flash_attn_ext`, both `build_self_attention` and `build_joint_attention` applied an extra `ggml_permute(ctx, attn_out, 0, 2, 1, 3)` before `ggml_cont` + `ggml_reshape_3d`. The GGML source (`ggml.c:5342-5343`) confirms flash attention already outputs `[head_dim, num_heads, seq, batch]` — exactly the shape needed to cleanly reshape to `[model_size, seq, batch]` by merging `head_dim` and `num_heads`.

The extra permute swapped `num_heads` and `seq`, producing `[head_dim, seq, num_heads, batch]`. After `cont` then `reshape` to `[model_size, seq, batch]`, each sequence position received head 0 from `num_heads` (16) different sequence positions instead of all 16 heads for that position. For example with head_dim=128, num_heads=16:
- Position 0 got head 0 data from seq positions 0–15 (16×128=2048 floats)
- Position 1 got head 0 data from seq positions 16–31

This feature scrambling propagated through all 24 decoder layers (and both 14-layer encoders), compounding to produce the ~1.84x output scale and only noise.

**Fix Applied:** Removed the `ggml_permute(ctx, attn_out, 0, 2, 1, 3)` line in both functions in `cpp/src/echo_model.cpp`:
- `build_self_attention()` — line 566 (affects encoder blocks)
- `build_joint_attention()` — line 754 (affects decoder blocks)

The flash attention output is already in the correct `[head_dim, num_heads, seq, batch]` layout — just `cont` + `reshape` directly.

**Root cause trace:** The GGML flash attention output shape is: `ne = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }`. Since Q is permuted to `[head_dim, seq, num_heads, batch]` before the call, the output is `[head_dim, num_heads, seq, batch]`. The comment "permuted" was misleading — the output is already in the desired contracted form.

---

## Other Known Differences (Not Bugs)

### Audio Resampling
- **Python:** `torchaudio.functional.resample` (sinc/Kaiser interpolation)
- **C++:** Simple linear interpolation in `cpp/src/echo_audio.cpp`
- **Impact:** speaker_zq correlation is 0.90 instead of 1.0. Speaker_zpca correlation is 0.84.
- **Severity:** Quality degradation, not a correctness bug. Could be improved with a proper sinc resampler.

### RNG Algorithm
- **Python:** `torch.Generator(device='cuda').manual_seed(seed)` — CUDA Philox RNG
- **C++:** `std::mt19937_64` — Mersenne Twister
- **Impact:** Different initial noise x_t (correlation ≈ 0). Both are valid Gaussian noise.
- **Severity:** Not a bug — just prevents exact reproducibility between Python and C++. For debugging, can load Python x_t into C++ via file.

### CFG Batching
- **Python:** Single forward pass with batch=3 (cond + 2 uncond variants)
- **C++:** Three separate forward passes with batch=1
- **Impact:** Mathematically equivalent. C++ approach is 3x slower but uses less memory.

---

## Key File Locations

### C++ Engine
- `cpp/src/echo_model.cpp` — GGML graph builder (1329 lines). **Most likely location of the 1.84x bug.**
  - `build_adaln()` — lines 533-584
  - `build_self_attention()` — lines 446-519
  - `build_joint_attention()` — lines 590-707
  - `build_decoder_block()` — lines 730-762
  - `forward()` — lines 1118-1329
  - `build_and_compute_kv_cache()` — lines 768-1077
- `cpp/src/echo_pipeline.cpp` — Pipeline orchestration, diagnostic dump
- `cpp/src/echo_sampler.cpp` — Euler sampler with 3-pass CFG
- `cpp/src/echo_tokenizer.cpp` — Byte-level tokenizer
- `cpp/src/echo_audio.cpp` — WAV I/O, resampling
- `cpp/src/echo_dac_onnx.cpp` — DAC encoder/decoder ONNX wrapper
- `cpp/src/echo_pca.cpp` — PCA encode/decode
- `cpp/src/echo_postprocess.cpp` — Silence detection & audio cropping

### Python Reference
- `model.py` — Full EchoDiT architecture (642 lines)
  - `LowRankAdaLN` — lines 46-83
  - `SelfAttention` — lines 106-161
  - `JointAttention` — lines 163-293
  - `EchoDiT.forward()` — lines 563-604
  - `EchoDiT.get_kv_cache_text()` — lines 606-613
  - `EchoDiT.get_kv_cache_speaker()` — lines 615-621
- `inference.py` — Sampling pipeline, audio loading, tokenizer
  - `sample_euler_cfg_independent_guidances()` — lines 341-431
  - `sample_pipeline()` — lines 253-302
- `gradio_app.py` — Full inference UI (working reference)

### Debug Scripts (in `scripts/`)
- `debug_dump.py` — Generate Python diagnostic dumps
- `compare_dumps.py` — Automated stage-by-stage comparison
- `analyze_diff.py` — Detailed numerical analysis
- `verify_weights.py` — GGUF vs PyTorch weight verification
- `dump_py_intermediates.py` — Save Python intermediate tensors
- `test_with_cpp_inputs.py` — Run Python model with C++ inputs
- `full_compare.py` — Comprehensive same-input comparison
- `layer_analysis.py` — Layer-by-layer std tracking

### Debug Data Directories
- `debug_py/` — Python diagnostic dumps (token_ids, speaker_zq, x_t, model_fwd_out, etc.)
- `debug_cpp/` — C++ diagnostic dumps (same stages)
- `debug_py_intermediates/` — Python layer-0 intermediate tensors
- `debug_py_from_cpp/` — Python model output using C++ inputs

---

## Build & Run Commands

```bash
# Build C++
cd cpp/build && cmake --build . --config Release

# Run C++ inference
cpp\build\Release\echo-tts.exe --model echo-dit.gguf --speaker kore_gemini.wav --text "Hello world" --dac-encoder onnx_models\dac_encoder.onnx --dac-decoder onnx_models\dac_decoder.onnx --seed 0

# Run C++ diagnostic dump
cpp\build\Release\echo-tts.exe --model echo-dit.gguf --speaker kore_gemini.wav --text "Hello world" --dac-encoder onnx_models\dac_encoder.onnx --dac-decoder onnx_models\dac_decoder.onnx --dump-intermediates debug_cpp --seed 0 --seq-length 640

# Run Python diagnostic dump
conda run -p .conda python scripts/debug_dump.py

# Compare dumps
conda run -p .conda python scripts/compare_dumps.py --py-dir debug_py --cpp-dir debug_cpp --thresh 1e-3

# Detailed analysis
conda run -p .conda python scripts/analyze_diff.py
```

---

## Model Architecture Summary

- **EchoDiT**: Diffusion transformer for TTS
  - Latent size: 80 (PCA-compressed from 1024-dim DAC codes)
  - Model size: 2048, 24 decoder layers, 16 heads, head_dim=128
  - Text encoder: 14 layers, 1280 dim, 10 heads (bidirectional)
  - Speaker encoder: 14 layers, 1280 dim, 10 heads (causal), patch_size=4
  - AdaLN: Low-rank (rank=256) with residual, conditioned on timestep
  - Decoder attention: Joint attention over [self, latent?, text, speaker] KV
  - Half-RoPE in decoder (first half of heads get RoPE, second half don't)
  - Full RoPE in encoders
  - Gated attention: `output * sigmoid(gate)`
  - SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`
- **DAC (S1-DAC)**: Autoencoder, 1024-dim latent, downsample factor 2048
- **PCA**: 1024 → 80 dim compression with latent_scale=0.0556
- **Sampling**: Euler ODE with independent text+speaker CFG, 40 steps, temporal score rescaling
