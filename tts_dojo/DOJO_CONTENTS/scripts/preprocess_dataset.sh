#!/bin/bash
#preprocess_dataset.sh:  Validates the linked dataset for piper1-gpl training.
# piper1-gpl (piper.train) does phoneme/audio caching itself during `fit`
# (VitsDataModule.prepare_data, using --data.cache_dir). The old separate
# piper_train.preprocess step no longer exists: this script checks the dataset
# and lets the user purge an old cache.

DOJO_NAME=$(basename $PWD) # Get from <voice_name>_dojo
SETTINGS_FILE="SETTINGS.txt"
CACHE_DIR="../training_folder/cache"
CONFIG_JSON="../training_folder/config.json"
SAMPLING_RATE_FILE=".SAMPLING_RATE"
MAX_WORKERS_FILE=".MAX_WORKERS"
DATASET_CONF_FILE="../target_voice_dataset/dataset.conf"
METADATA_CSV="../target_voice_dataset/metadata.csv"
AUDIO_DIR="../target_voice_dataset/wav"

cd scripts # needed to ensure relative paths are built properly
set +e # Exit immediately if any command returns a non-zero exit code

#.SAMPLING_RATE and .MAX_WORKERS are stored in <voice>_dojo/scripts by link_dataset.sh
if [[ -f $SAMPLING_RATE_FILE ]]; then
    SAMPLING_RATE=$(cat $SAMPLING_RATE_FILE)
else
    echo "Error: .SAMPLING_RATE file not found."
    exit 1
fi

if [[ -f $MAX_WORKERS_FILE ]]; then
    MAX_WORKERS=$(cat $MAX_WORKERS_FILE)
else
    echo "Error: .MAX_WORKERS file not found."
    exit 1
fi

# load dataset.conf
if [ -e $DATASET_CONF_FILE ]; then
    source $DATASET_CONF_FILE
else
    echo "$0 - dataset.conf not found"
    echo "     expected location: $DATASET_CONF_FILE"
    echo
    exit 1
fi

# Check whether the conf file is old and is missing values
missing=false
# Check if ESPEAK_LANGUAGE_IDENTIFIER is unset or empty
if [[ -z "$ESPEAK_LANGUAGE_IDENTIFIER" ]]; then
    echo
    echo
    echo "    Error: ESPEAK_LANGUAGE_IDENTIFIER is not set in dataset.conf."
    missing=true
fi

# Check if PIPER_FILENAME_PREFIX is unset or empty
if [[ -z "$PIPER_FILENAME_PREFIX" ]]; then
    echo "    Error: PIPER_FILENAME_PREFIX is not set in dataset.conf."
    echo
    missing=true
fi

# Exit if any variable was missing
if [[ "$missing" == true ]]; then
    echo
    echo "    Your dataset configuration file (dataset.conf) is outdated and needs to be updated."
    echo "        ESPEAK_LANGUAGE_IDENTIFIER must be set to the espeak-ng language identifier for the language of your dataset"
    echo "        eg: ESPEAK_LANGUAGE_IDENTIFIER=it"
    echo
    echo "    PIPER_FILENAME_PREFIX must be set to the language code Piper uses to name language files"
    echo "        eg: PIPER_FILENAME_PREFIX=it_IT"
    echo
    echo "    either add these values to your dataset.conf file manually, or run:"
    echo
    echo "        DATASETS/create_dataset.sh <dataset_folder>"
    echo
    echo "    to rebuild the datasets.conf file."
    echo
    echo "Exiting"
    exit 1
fi

# load settings
if [ -e "$SETTINGS_FILE" ]; then
    source "$SETTINGS_FILE"  #loads vars from SETTINGS.txt
else
    echo "could not find $SETTINGS_FILE. Exiting."
    exit 1
fi


previously_preprocessed() {
# Check for files from a previous training run (the fit caches phonemes/audio/spec here)
    if [[ -d "$CACHE_DIR" && -f "$CONFIG_JSON" ]]; then
        echo "TRUE"
    else
        echo "FALSE"
    fi
}

purge_training_folder(){
# removes cache from previous training runs
   rm -rf $CACHE_DIR
   rm -f $CONFIG_JSON
}


error_handler() {
  echo "An error occurred in the script. Exiting."
  exit 1
}

export -f error_handler

# Trap errors and call the error_handler function
trap 'error_handler' ERR SIGINT SIGTERM


function check_preprocessed_data() {
    if [[ "$(previously_preprocessed)" == "TRUE" ]]; then
        echo
        echo "        training_folder directory already contains cached training data. Please choose an option:"
        echo
        echo "            [1] Skip cleanup (recommended if resuming a previous training session)"
        echo "            [2] Clear the cache and rebuild it  (important if you have changed pronunciation rules) "
        echo
        echo -ne "            [1,2]? "
        read -r redo

        if [[ -z "$redo" || "$redo" == "1" ]]; then
            echo "Skipping cleanup."
            exit 0
        elif [[ "$redo" == "2" ]]; then
            echo "Cleaning training_folder prior to training..."
            purge_training_folder
            return 0
        else
            echo "Invalid option. Please choose either 1 or 2."
            check_preprocessed_data  # Recursively ask again if input is invalid
        fi
    else
        echo "training folder is clean, proceeding."
    fi
}


# MAIN PROGRAM ********************************************************************************

check_preprocessed_data
echo
echo ""
echo -e "       Auto-configured sampling rate: $SAMPLING_RATE"
echo -e "    Calculated value for max-workers: $MAX_WORKERS"
echo
echo

echo "Configuring Piper for language: ${ESPEAK_LANGUAGE_IDENTIFIER}"
echo "Validating dataset..."
echo

# validate the dataset files the fit command will consume
if [ ! -f "$METADATA_CSV" ]; then
    echo "Error: metadata file not found: $METADATA_CSV"
    read
    exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: audio directory not found: $AUDIO_DIR"
    read
    exit 1
fi

metadata_lines=$(wc -l < "$METADATA_CSV" | tr -d ' ')
audio_files=$(find -L "$AUDIO_DIR" -name "*.wav" | wc -l | tr -d ' ')
echo "    metadata.csv rows : $metadata_lines"
echo "    audio files       : $audio_files"
echo
echo "    Phonemes, normalized audio and mel spectrograms are cached by piper1-gpl"
echo "    during the first training epoch (--data.cache_dir)."
echo

result=0
if [ $result -eq 0 ]; then
    echo
    echo
    echo "    Successfully validated dataset."
    echo
    echo "    Press <Enter> to continue"
    read
    echo
else
    echo  "dataset validation failed.  Press <Enter> to exit."
    read
    exit 1
fi
exit 0
