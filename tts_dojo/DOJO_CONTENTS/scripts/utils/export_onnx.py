#!/usr/bin/env python3
"""Export wrapper for piper1-gpl (piper.train.export_onnx).

Handles the two torch 2.6+/2.7+ compatibility issues:
  - weights_only=True default: allowlist pathlib globals found in legacy checkpoints
  - dynamo-based default exporter fails on VITS: force the legacy (TorchScript) exporter

Usage: python3 export_onnx.py --checkpoint <ckpt> --output-file <out.onnx>
"""
import pathlib
import sys

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

_orig_export = torch.onnx.export


def _legacy_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _orig_export(*args, **kwargs)


torch.onnx.export = _legacy_export

from piper.train.export_onnx import main  # noqa: E402

main()
