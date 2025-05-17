#!/bin/bash

# Prompt for data name
read -p "Enter data name (e.g., galaxies): " data_name

read -p "Enter data path (e.g., Data/galaxies/): " input_path

# Prompt for mode (train/predict/all)
read -p "Enter mode (train/predict/all/fresh): " mode

# Prompt for encoder type (sutoenc/cnn)
read -p "Enter encoder type for cdbnn (autoenc/cnn): " encoder_type

# Prompt for model type (Histogram/Gaussian)
read -p "Enter model type for adbnn (Histogram/Gaussian): " model


# Git pull
git pull

# Run operations based on mode
case $mode in
  "fresh")
    echo "Cleaning..."
    rm -rf "data/${data_name}/"
    rm -rf Model/*"${data_name}"_*
    rm -rf training_data/${data_name}/
    ;;&
  "train" | "all"| "fresh")

    echo "Running training..."
    python cdbnn1.py --mode train --data_name "$data_name" --input_path "$input_path"  #--encoder_type "$encoder_type"
    python cdbnn1.py --mode predict --data_name "$data_name" --input_path "$input_path" #--encoder_type "$encoder_type"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode train --model_type "$model"
    ;;&  # Continue to next case (executes predict if mode is "all")
  "predict" | "all" | "fresh")
    echo "Running prediction..."
    python cdbnn1.py --mode predict --data_name "$data_name" --input_path "$input_path" #--encoder_type "$encoder_type"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode predict --model_type "$model"
    ;;
  *)
    echo "Invalid mode. Choose 'train', 'predict', or 'all'"
    exit 1
    ;;
esac

echo "Pipeline completed!"
