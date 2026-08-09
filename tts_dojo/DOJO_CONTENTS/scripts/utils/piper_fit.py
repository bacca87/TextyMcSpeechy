#!/usr/bin/env python3
"""piper_fit.py — dojo training launcher for piper1-gpl.

Runs `python -m piper.train fit` semantics via the Python API, which avoids the
LightningCLI quirks (YAML-style arg parsing, legacy checkpoint hparams rejection)
and works with BOTH legacy (piper_train, epoch=N-step=M) and new checkpoints.

Reads a JSON parameter file (written by utils/piper_training.sh).

Usage: python3 piper_fit.py <params.json>
"""
import json
import os
import pathlib
import re
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

CKPT = p.get("ckpt_path")
FINE_TUNE_LR = p.get("learning_rate")
if FINE_TUNE_LR:
    # Fine-tuning mode: load ALL weights from the checkpoint but start a FRESH
    # optimizer/scheduler with the explicit learning rate from SETTINGS.txt.
    # piper1-gpl's warmstart_ckpt copies every matching-shape parameter and
    # leaves the optimizer fresh -- required when resuming from a checkpoint
    # whose LR scheduler is already decayed (eg a finished training run).
    print("Fine-tuning weights from checkpoint (fresh optimizer, "
          f"lr_g={FINE_TUNE_LR}):", CKPT)
    model = VitsModel(**MODEL_ARGS, learning_rate=float(FINE_TUNE_LR),
                      learning_rate_d=float(FINE_TUNE_LR) / 2,
                      warmstart_ckpt=CKPT)
    resume_ckpt = None
elif CKPT:
    # New-format checkpoints (epoch=N-val_mel=.../epoch=N-val_mos=...) are
    # Lightning 2.x checkpoints: hand them to trainer.fit(ckpt_path=...) so
    # training resumes at the saved epoch with optimizer/scheduler state.
    if re.search(r"val_(?:mel|mos)=", os.path.basename(CKPT)):
        print("Resuming from checkpoint (full state):", CKPT)
        model = VitsModel(**MODEL_ARGS)
        resume_ckpt = CKPT
    else:
        # Legacy piper_train checkpoints (epoch=N-step=M, Lightning 1.x) have
        # no compatible optimizer state: load weights only and restart fresh.
        print("Resuming weights from legacy checkpoint:", CKPT)
        model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu")
        resume_ckpt = None
else:
    print("Training from scratch")
    model = VitsModel(**MODEL_ARGS)
    resume_ckpt = None

trainer = L.Trainer(
    max_epochs=int(p.get("max_epochs", 30000)),
    accelerator="gpu",
    devices=1,
    precision=32,
    callbacks=_DEFAULT_CALLBACKS,
    default_root_dir=p["training_dir"],
)
trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
