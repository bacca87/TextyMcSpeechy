#!/usr/bin/env python3
"""piper_fit.py — dojo training launcher for piper1-gpl.

Runs `python -m piper.train fit` semantics via the Python API, which avoids the
LightningCLI quirks (YAML-style arg parsing, legacy checkpoint hparams rejection)
and works with BOTH legacy (piper_train, epoch=N-step=M) and new checkpoints.

Reads a JSON parameter file (written by utils/piper_training.sh).

Usage: python3 piper_fit.py <params.json>
"""
import json
import pathlib
import sys

import lightning as L
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

from piper.train.__main__ import _DEFAULT_CALLBACKS
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel

with open(sys.argv[1], encoding="utf-8") as f:
    p = json.load(f)

QUALITY = p["quality"]  # L | M | H

MODEL_ARGS = dict(
    sample_rate=int(p["sample_rate"]),
    batch_size=int(p["batch_size"]),
)
if QUALITY == "L":
    MODEL_ARGS.update(hidden_channels=96, inter_channels=96, filter_channels=384)
elif QUALITY == "H":
    MODEL_ARGS.update(
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
    )

dm = VitsDataModule(
    csv_path=p["csv_path"],
    cache_dir=p["cache_dir"],
    espeak_voice=p["espeak_voice"],
    config_path=p["config_path"],
    voice_name=p["voice_name"],
    sample_rate=int(p["sample_rate"]),
    audio_dir=p["audio_dir"],
    batch_size=int(p["batch_size"]),
    num_workers=int(p["num_workers"]),
    validation_split=float(p["validation_split"]),
)

if p.get("ckpt_path"):
    print("Resuming from checkpoint:", p["ckpt_path"])
    model = VitsModel.load_from_checkpoint(p["ckpt_path"], map_location="cpu")
else:
    print("Training from scratch")
    model = VitsModel(**MODEL_ARGS)

trainer = L.Trainer(
    max_epochs=int(p.get("max_epochs", 30000)),
    accelerator="gpu",
    devices=1,
    precision=32,
    callbacks=_DEFAULT_CALLBACKS,
    default_root_dir=p["training_dir"],
)
trainer.fit(model, datamodule=dm)
