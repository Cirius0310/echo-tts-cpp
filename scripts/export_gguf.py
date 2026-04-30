"""
Export EchoDiT model weights + PCA state to GGUF format for C++ inference.

Usage:
    python export_gguf.py [--repo-id jordand/echo-tts-base] [--output echo-dit.gguf] [--dtype f16]

Requires: pip install gguf safetensors huggingface-hub torch
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("ERROR: `gguf` package not found. Install with: pip install gguf")
    sys.exit(1)

from huggingface_hub import hf_hub_download


# ── Hyperparameters (must match model.py EchoDiT constructor) ────────
HPARAMS = {
    "latent_size": 80,
    "model_size": 2048,
    "num_layers": 24,
    "num_heads": 16,
    "intermediate_size": 5888,
    "norm_eps": 1e-5,
    "text_vocab_size": 256,
    "text_model_size": 1280,
    "text_num_layers": 14,
    "text_num_heads": 10,
    "text_intermediate_size": 3328,
    "speaker_patch_size": 4,
    "speaker_model_size": 1280,
    "speaker_num_layers": 14,
    "speaker_num_heads": 10,
    "speaker_intermediate_size": 3328,
    "timestep_embed_size": 512,
    "adaln_rank": 256,
}


def get_ggml_type(dtype_str: str) -> GGMLQuantizationType:
    mapping = {
        "f32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "bf16": GGMLQuantizationType.BF16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(mapping.keys())}")
    return mapping[dtype_str]


def convert_tensor(tensor: torch.Tensor, target_dtype: str) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array in the target dtype."""
    # Always go through float32 for safety, then convert
    t = tensor.detach().cpu().float()
    if target_dtype == "f32":
        return t.numpy()
    elif target_dtype == "f16":
        return t.half().numpy()
    elif target_dtype == "bf16":
        return t.to(torch.bfloat16).view(torch.int16).numpy()
    else:
        return t.numpy()


def main():
    parser = argparse.ArgumentParser(description="Export EchoDiT to GGUF")
    parser.add_argument("--repo-id", default="jordand/echo-tts-base",
                        help="HuggingFace repo ID for EchoDiT weights")
    parser.add_argument("--output", default="echo-dit.gguf",
                        help="Output GGUF filename")
    parser.add_argument("--dtype", default="f16", choices=["f32", "f16", "bf16"],
                        help="Storage dtype for weight tensors")
    parser.add_argument("--include-blockwise", action="store_true", default=True,
                        help="Include blockwise/latent encoder weights")
    parser.add_argument("--local-weights", default=None,
                        help="Path to local safetensors file (skip HF download)")
    parser.add_argument("--local-pca", default=None,
                        help="Path to local PCA state safetensors file")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token")
    args = parser.parse_args()

    # ── Load weights ──
    if args.local_weights:
        w_path = args.local_weights
    else:
        print(f"Downloading model weights from {args.repo_id}...")
        w_path = hf_hub_download(args.repo_id, "pytorch_model.safetensors", token=args.token)

    print(f"Loading weights from {w_path}...")
    state = st.load_file(w_path, device="cpu")

    # Optionally filter out blockwise modules
    if not args.include_blockwise:
        state = {k: v for k, v in state.items() if not (
            k.startswith("latent_encoder.") or
            k.startswith("latent_norm") or
            ".wk_latent" in k or
            ".wv_latent" in k
        )}

    # ── Load PCA state ──
    if args.local_pca:
        p_path = args.local_pca
    else:
        print(f"Downloading PCA state from {args.repo_id}...")
        p_path = hf_hub_download(args.repo_id, "pca_state.safetensors", token=args.token)

    print(f"Loading PCA state from {p_path}...")
    pca_state = st.load_file(p_path, device="cpu")

    # ── Write GGUF ──
    ggml_dtype = get_ggml_type(args.dtype)
    writer = GGUFWriter(args.output, arch="echo-tts")

    # Write hyperparameters as metadata
    print("Writing hyperparameters...")
    for key, value in HPARAMS.items():
        if isinstance(value, int):
            writer.add_uint32(f"echo.{key}", value)
        elif isinstance(value, float):
            writer.add_float32(f"echo.{key}", value)

    writer.add_bool("echo.include_blockwise", args.include_blockwise)
    writer.add_string("echo.dtype", args.dtype)

    # Write PCA state as f32 (small tensors, need precision)
    print("Writing PCA state tensors...")
    pca_components = pca_state["pca_components"].float().numpy()  # (80, 1024)
    pca_mean = pca_state["pca_mean"].float().numpy()              # (1024,)
    latent_scale = float(pca_state["latent_scale"].item())

    writer.add_tensor("pca.components", pca_components, raw_dtype=GGMLQuantizationType.F32)
    writer.add_tensor("pca.mean", pca_mean, raw_dtype=GGMLQuantizationType.F32)
    writer.add_float32("pca.latent_scale", latent_scale)

    # Write model weights
    n_tensors = len(state)
    print(f"Writing {n_tensors} model tensors as {args.dtype}...")

    for i, (name, tensor) in enumerate(state.items()):
        if (i + 1) % 100 == 0 or (i + 1) == n_tensors:
            print(f"  [{i+1}/{n_tensors}] {name} {list(tensor.shape)}")

        arr = convert_tensor(tensor, args.dtype)

        # Use f32 for small tensors (norms, biases, embeddings < 1M elements)
        # to preserve precision where it matters
        use_f32 = tensor.numel() < 1_000_000
        if use_f32:
            arr_f32 = tensor.detach().cpu().float().numpy()
            writer.add_tensor(name, arr_f32, raw_dtype=GGMLQuantizationType.F32)
        else:
            writer.add_tensor(name, arr, raw_dtype=ggml_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! Wrote {output_path} ({size_mb:.1f} MB)")
    print(f"  {n_tensors} model tensors + 2 PCA tensors")
    print(f"  Dtype: {args.dtype} (small tensors kept as f32)")


if __name__ == "__main__":
    main()
