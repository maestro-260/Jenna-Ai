#!/bin/bash

# Define dataset path
DATASET_PATH="training"

# Run data preparation step
python -m training.data_prep

# Start fine-tuning
ollama run mistral --train-file "$DATASET_PATH/train.jsonl" --val-file "$DATASET_PATH/val.jsonl"
