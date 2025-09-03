#!/bin/bash

# YOLOv11 Coin Detection Training Script
# Usage: bash train_coin.sh [num_gpus]
# Example: bash train_coin.sh 1  (for single GPU)
#          bash train_coin.sh 2  (for 2 GPUs)

set -e  # Exit on any error

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated. Please run:"
    echo "   source .venv/bin/activate"
    exit 1
fi

# Get number of GPUs (default to 1)
NUM_GPUS=${1:-1}

echo "🚀 Starting YOLOv11 Coin Detection Training..."
echo "📊 Number of GPUs: $NUM_GPUS"

# Check if main_coin.py exists
if [ ! -f "main_coin.py" ]; then
    echo "❌ main_coin.py not found in current directory"
    echo "   Make sure you're running this script from the yolov11 directory"
    exit 1
fi

# Check if coin_args.yaml exists
if [ ! -f "utils/coin_args.yaml" ]; then
    echo "❌ utils/coin_args.yaml not found"
    echo "   Make sure the configuration file exists"
    exit 1
fi

# Create weights directory if it doesn't exist
mkdir -p weights

echo "🔧 Training configuration:"
echo "   - Script: main_coin.py"
echo "   - Config: utils/coin_args.yaml"
echo "   - GPUs: $NUM_GPUS"
echo "   - Output: weights/"
echo ""

# Run training based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "🎯 Starting single GPU training..."
    python3 main_coin.py --train --batch-size 16 --epochs 100
elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "🎯 Starting multi-GPU training with $NUM_GPUS GPUs..."
    python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main_coin.py --train --batch-size 16 --epochs 100
else
    echo "❌ Invalid number of GPUs: $NUM_GPUS"
    echo "   Please provide a positive integer (1, 2, 3, etc.)"
    exit 1
fi

echo ""
echo "✅ Training completed!"
echo "📁 Check the following files for results:"
echo "   - weights/best.pt (best model)"
echo "   - weights/last.pt (last epoch model)"
echo "   - weights/step.csv (training metrics)"
echo ""
echo "🧪 To test the trained model, run:"
echo "   python3 main_coin.py --test"
