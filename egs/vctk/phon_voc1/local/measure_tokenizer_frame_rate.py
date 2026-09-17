#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Empirically measure the Sony Phonological-Tokenizer's output frame rate.

Usage:
    python measure_tokenizer_frame_rate.py --tokenizer-dir downloads/phonological_tokenizer
"""

import argparse
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=str, required=True)
    parser.add_argument("--duration", type=float, default=3.03)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, args.tokenizer_dir)
    from tokenizer import PhonologicalTokenizer  # NOQA

    sr = 16000
    n = int(sr * args.duration)
    t = np.arange(n) / sr
    wav = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav_path = "/tmp/_tokenizer_frame_rate_test.wav"
    sf.write(wav_path, wav, sr)

    tokenizer = PhonologicalTokenizer(
        ssl_model_path=f"{args.tokenizer_dir}/ssl.pth",
        centroids_path=f"{args.tokenizer_dir}/centroids.npy",
        device=args.device,
    )
    with torch.no_grad():
        clusters = tokenizer(wav_path)

    num_tokens = clusters.shape[-1]
    print(f"input samples:      {n}")
    print(f"input duration (s): {n / sr:.4f}")
    print(f"num tokens:         {num_tokens}")
    print(f"implied hop (samples/token @16kHz): {n / num_tokens:.3f}")
    print(f"implied frame period (ms):          {1000 * n / num_tokens / sr:.3f}")
    print(f"vocab size (centroids.npy rows):    {tokenizer.centroids.shape[0]}")
    print(f"token value range: [{clusters.min().item()}, {clusters.max().item()}]")


if __name__ == "__main__":
    main()
