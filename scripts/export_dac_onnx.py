"""
Export Fish Speech S1-DAC autoencoder to ONNX for C++ inference.

Exports two models:
  1. dac_encoder.onnx  — audio waveform → z_q (pre-PCA continuous latent)
  2. dac_decoder.onnx  — z_q (pre-PCA continuous latent) → audio waveform

Usage:
    python export_dac_onnx.py --echo-tts-path ../echo-tts [--output-dir ./onnx_models]

Requires: pip install torch onnx onnxruntime safetensors huggingface-hub einops
"""

import argparse
import sys
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np


class DACEncoder(nn.Module):
    """Wrapper that exposes S1-DAC encode_zq as a simple forward()."""

    def __init__(self, dac):
        super().__init__()
        self.dac = dac

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (batch, 1, length) — mono waveform at 44100Hz
        Returns:
            z_q: (batch, channels, time) — continuous latent, channels=1024
        """
        return self.dac.encode_zq(audio)


class DACDecoder(nn.Module):
    """Wrapper that exposes S1-DAC decode_zq as a simple forward()."""

    def __init__(self, dac):
        super().__init__()
        self.dac = dac

    @torch.no_grad()
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (batch, channels, time) — continuous latent, channels=1024
        Returns:
            audio: (batch, 1, length) — decoded audio at 44100Hz
        """
        return self.dac.decode_zq(z_q)


def export_encoder(dac, output_path: str, opset_version: int = 17):
    """Export the DAC encoder to ONNX."""
    print(f"Exporting encoder to {output_path}...")

    encoder = DACEncoder(dac).eval()

    # Create dummy input: ~1 second of audio at 44100Hz
    # Length must be divisible by hop_length * 4 = 512 * 4 = 2048
    dummy_length = 2048 * 32  # ~1.5 seconds
    dummy_audio = torch.randn(1, 1, dummy_length, dtype=dac.dtype, device=dac.device)

    # Test forward pass
    with torch.no_grad():
        test_output = encoder(dummy_audio)
    print(f"  Encoder test: input {list(dummy_audio.shape)} → output {list(test_output.shape)}")

    # Move to CPU for export
    encoder = encoder.cpu().float()
    dummy_audio = dummy_audio.cpu().float()

    torch.onnx.export(
        encoder,
        dummy_audio,
        output_path,
        opset_version=opset_version,
        input_names=["audio"],
        output_names=["z_q"],
        dynamic_axes={
            "audio": {0: "batch", 2: "length"},
            "z_q": {0: "batch", 2: "time"},
        },
    )
    print(f"  ✓ Encoder exported ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")


def export_decoder(dac, output_path: str, opset_version: int = 17):
    """Export the DAC decoder to ONNX."""
    print(f"Exporting decoder to {output_path}...")

    decoder = DACDecoder(dac).eval()

    # Create dummy z_q input
    # z_q shape is (batch, 1024, time) where time corresponds to encoded audio length
    dummy_time = 32  # ~1.5 seconds worth of latents
    dummy_zq = torch.randn(1, 1024, dummy_time, dtype=dac.dtype, device=dac.device)

    # Test forward pass
    with torch.no_grad():
        test_output = decoder(dummy_zq)
    print(f"  Decoder test: input {list(dummy_zq.shape)} → output {list(test_output.shape)}")

    # Move to CPU for export
    decoder = decoder.cpu().float()
    dummy_zq = dummy_zq.cpu().float()

    torch.onnx.export(
        decoder,
        dummy_zq,
        output_path,
        opset_version=opset_version,
        input_names=["z_q"],
        output_names=["audio"],
        dynamic_axes={
            "z_q": {0: "batch", 2: "time"},
            "audio": {0: "batch", 2: "length"},
        },
    )
    print(f"  ✓ Decoder exported ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")


def verify_onnx(onnx_path: str, input_name: str, dummy_input: np.ndarray):
    """Verify ONNX model loads and runs correctly."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        result = sess.run(None, {input_name: dummy_input})
        print(f"  ✓ Verification passed: output shape {result[0].shape}")
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export S1-DAC to ONNX")
    parser.add_argument("--echo-tts-path", required=True,
                        help="Path to the original echo-tts Python repo (contains autoencoder.py, inference.py)")
    parser.add_argument("--output-dir", default="./onnx_models",
                        help="Output directory for ONNX files")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="Model dtype for loading (export is always float32)")
    parser.add_argument("--device", default="cuda",
                        help="Device for loading model")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify exported models with ONNX Runtime")
    parser.add_argument("--skip-encoder", action="store_true",
                        help="Skip encoder export (only export decoder)")
    args = parser.parse_args()

    echo_tts_dir = Path(args.echo_tts_path).resolve()
    if not echo_tts_dir.is_dir():
        print(f"ERROR: echo-tts Python repo not found at {echo_tts_dir}")
        sys.exit(1)

    sys.path.insert(0, str(echo_tts_dir))
    from autoencoder import DAC, build_ae
    from inference import load_fish_ae_from_hf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Loading S1-DAC model (dtype={args.dtype}, device={args.device})...")
    fish_ae = load_fish_ae_from_hf(device=args.device, dtype=dtype)
    print(f"  Model loaded. hop_length={fish_ae.hop_length}, sample_rate={fish_ae.sample_rate}")

    # Export encoder
    if not args.skip_encoder:
        encoder_path = str(output_dir / "dac_encoder.onnx")
        export_encoder(fish_ae, encoder_path, args.opset)

        if args.verify:
            dummy = np.random.randn(1, 1, 2048 * 32).astype(np.float32)
            verify_onnx(encoder_path, "audio", dummy)

    # Export decoder
    decoder_path = str(output_dir / "dac_decoder.onnx")
    export_decoder(fish_ae, decoder_path, args.opset)

    if args.verify:
        dummy = np.random.randn(1, 1024, 32).astype(np.float32)
        verify_onnx(decoder_path, "z_q", dummy)

    print(f"\nDone! ONNX models saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.glob("*.onnx")):
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
