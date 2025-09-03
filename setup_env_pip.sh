#!/bin/bash

# YOLOv11 Coin Detection Environment Setup Script
# This script creates and configures a Python virtual environment for YOLOv11 coin detection

set -e  # Exit on any error

echo "🚀 Setting up YOLOv11 Coin Detection Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Python version: $PYTHON_VERSION"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment
echo "📦 Creating virtual environment '.venv'..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust index URL based on your CUDA version)
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "📚 Installing project requirements..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To start training, run:"
echo "  bash train_coin.sh 1"
