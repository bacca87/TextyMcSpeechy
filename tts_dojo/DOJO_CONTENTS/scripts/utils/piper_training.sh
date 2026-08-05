#!/bin/bash
# scripts/piper_training.sh - Runs training inside docker container (piper1-gpl).
# Expects one parameter:  a path (container-side) to a starting checkpoint file
echo "Running piper_training.sh"

trap "kill 0" SIGINT
set +e # Exit immediately if any command returns a non-zero exit code

SETTINGS_FILE="SETTINGS.txt"

# flag file that overrides the use of pretrained checkpoints
TRAIN_FROM_SCRATCH_FILE="../target_voice_dataset/.SCRATCH"

# file containing quality setting for this dojo set by link_dataset.sh
QUALITY_FILE="../target_voice_dataset/.QUALITY"

# dataset configuration (NAME, DESCRIPTION, ESPEAK_LANGUAGE_IDENTIFIER, ...)
DATASET_CONF_FILE="../target_voice_dataset/dataset.conf"

# sampling rate and dataloader worker count written by the dataset tools
SAMPLING_RATE_FILE=".SAMPLING_RATE"
MAX_WORKERS_FILE=".MAX_WORKERS"

# infer name of dojo and voice from directory name
DOJO_NAME=$(basename "$(dirname "$PWD")")  # this script runs from <name>_dojo/scripts so need parent directory
VOICE_NAME=$(echo "$DOJO_NAME" | sed 's/_dojo$//')

# sanity check for current directory
if [[ ! "$DOJO_NAME" =~ _dojo$ ]]; then
    echo "Error: DOJO_NAME did not end with '_dojo'. Are you running from <voice>_dojo/scripts directory?  Exiting." >&2
    exit 1
fi

# load training settings from SETTINGS_FILE
if [ -e $SETTINGS_FILE ]; then
    source $SETTINGS_FILE
else
    echo "$0 - settings not found"
    echo "     expected location: $SETTINGS_FILE"
    echo
    echo "press <enter> to exit"
    exit 1
fi

if [[ -f $TRAIN_FROM_SCRATCH_FILE ]]; then
    TRAIN_FROM_SCRATCH=$(cat $TRAIN_FROM_SCRATCH_FILE)
else
    echo "Error: .SCRATCH file not found: $TRAIN_FROM_SCRATCH_FILE ."
    exit 1
fi

if [ ! -e "$QUALITY_FILE" ]; then
    echo "        Unable to proceed - file missing: $QUALITY_FILE"
    echo "        Please reconfigure this dojo's dataset.  Exiting."
    exit 1
fi

if [ ! -e "$DATASET_CONF_FILE" ]; then
    echo "        Unable to proceed - file missing: $DATASET_CONF_FILE"
    echo "        Please reconfigure this dojo's dataset.  Exiting."
    exit 1
fi

quality=$(cat $QUALITY_FILE)
if [ "$quality" = "L" ]; then
    quality_str="x-low"
elif [ "$quality" = "M" ]; then
    quality_str="medium"
elif [ "$quality" = "H" ]; then
    quality_str="high"
else
    echo "ERROR: $QUALITY_FILE contents invalid.  Exiting."
    exit 1
fi

source $DATASET_CONF_FILE

if [ -e "$SAMPLING_RATE_FILE" ]; then
    sample_rate=$(cat $SAMPLING_RATE_FILE)
else
    sample_rate=22050
fi

if [ -e "$MAX_WORKERS_FILE" ]; then
    num_workers=$(cat $MAX_WORKERS_FILE)
else
    num_workers=8
fi

if [ -z "$1" ]; then
  echo "No starting checkpoint received."
fi

starting_checkpoint=$1

if [ "$TRAIN_FROM_SCRATCH" = "true" ]; then
    echo "Training model from scratch (ignoring any starting checkpoint)."
    starting_checkpoint=""
fi

TRAINING_DIR="/app/tts_dojo/$DOJO_NAME/training_folder"
DATASET_DIR="/app/tts_dojo/$DOJO_NAME/target_voice_dataset"

# cache location: CACHE_DIR from SETTINGS.txt (default ../training_folder/cache,
# relative to the dojo) or an absolute path (e.g. a fast docker volume)
CACHE_DIR=${CACHE_DIR:-../training_folder/cache}
case "$CACHE_DIR" in
    /*) DOCKER_CACHE_DIR="$CACHE_DIR" ;;
    *)  DOCKER_CACHE_DIR="/app/tts_dojo/$DOJO_NAME/${CACHE_DIR#../}" ;;
esac

echo "Train from scratch = $TRAIN_FROM_SCRATCH"
echo "           Quality = $quality_str"
echo "   espeak voice    = $ESPEAK_LANGUAGE_IDENTIFIER"
echo "   sample rate     = $sample_rate"

# write the fit parameters (read by utils/piper_fit.py inside the container)
cat > ../training_folder/fit_params.json <<EOF
{
    "voice_name": "$VOICE_NAME",
    "csv_path": "$DATASET_DIR/metadata.csv",
    "audio_dir": "$DATASET_DIR/wav",
    "cache_dir": "$DOCKER_CACHE_DIR",
    "config_path": "$TRAINING_DIR/config.json",
    "espeak_voice": "$ESPEAK_LANGUAGE_IDENTIFIER",
    "sample_rate": "$sample_rate",
    "batch_size": "$PIPER_BATCH_SIZE",
    "num_workers": "$num_workers",
    "validation_split": "$VALIDATION_SPLIT",
    "quality": "$quality",
    "training_dir": "$TRAINING_DIR",
    "max_epochs": "30000",
    "ckpt_path": "$starting_checkpoint"
}
EOF

docker exec textymcspeechy-piper python3 "/app/tts_dojo/$DOJO_NAME/scripts/utils/piper_fit.py" "$TRAINING_DIR/fit_params.json"

exit 0
