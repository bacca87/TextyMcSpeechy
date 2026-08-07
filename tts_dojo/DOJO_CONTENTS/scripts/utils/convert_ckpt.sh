#!/bin/bash
# utils/convert_ckpt.sh — converts a legacy piper_train checkpoint (Lightning 1.x, epoch=N-step=M)
# to the piper1-gpl format so it can be resumed with `python -m piper.train fit --ckpt_path`.
# Runs INSIDE the docker container (paths are container paths).
# Usage: bash utils/convert_ckpt.sh <old_ckpt> <new_ckpt> <quality:L|M|H>
set -e

OLD_CKPT=$1
NEW_CKPT=$2
QUALITY=$3

if [ "$QUALITY" = "L" ]; then
    MODEL_ARGS="hidden_channels=96 inter_channels=96 filter_channels=384"
elif [ "$QUALITY" = "H" ]; then
    MODEL_ARGS="resblock=1 resblock_kernel_sizes=[3,7,11] resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]] upsample_rates=[8,8,2,2] upsample_initial_channel=512 upsample_kernel_sizes=[16,16,4,4]"
else
    MODEL_ARGS=""
fi

cat > /tmp/convert_ckpt.py <<PYEOF
import ast
import pathlib
import sys

import lightning as L
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

from piper.train.vits.lightning import VitsModel

OLD = sys.argv[1]
NEW = sys.argv[2]


def parse_value(s):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


# load legacy checkpoint with the new model class (Lightning auto-upgrades 1.x -> 2.x)
legacy = VitsModel.load_from_checkpoint(OLD, map_location="cpu")
print("legacy ckpt loaded; hparams:", sorted(legacy.hparams.keys())[:6], "...")

# rebuild the same architecture with explicit piper1-gpl hyperparameters
model_args = dict(
    sample_rate=int(legacy.hparams.sample_rate),
    num_symbols=int(legacy.hparams.num_symbols),
    num_speakers=int(legacy.hparams.num_speakers),
    batch_size=int(legacy.hparams.batch_size),
)
EXTRA = sys.argv[3]
for kv in EXTRA.split():
    k, v = kv.split("=", 1)
    if k == "resblock":
        # resblock is a string parameter ("1"/"2") in VitsModel; keep it a string
        model_args[k] = "1"
    else:
        model_args[k] = parse_value(v)

model = VitsModel(**model_args)
model.load_state_dict(legacy.state_dict())

# save in the piper1-gpl checkpoint format (new-style hyperparameters)
torch.save(
    {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": L.__version__,
    },
    NEW,
)
print("converted checkpoint written:", NEW)
PYEOF

/app/piper/.venv/bin/python /tmp/convert_ckpt.py "$OLD_CKPT" "$NEW_CKPT" "$MODEL_ARGS"
