# YOLOv11 Coin Detection Training

This directory contains the setup for training YOLOv11 on a coin detection dataset.

## Environment Setup

Before starting training, you need to set up a Python virtual environment with the required dependencies.

1. ### Automated Setup (Recommended)

```bash
# Create and setup pip virtual environment
bash setup_env_pip.sh

# Activate environment
source .venv/bin/activate
```

2. ### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

**Note**: The `setup_env_pip.sh` script is located in the project root directory. Make sure to run it from there, or adjust the path accordingly.


3. **Start Training**:
   ```bash
   # Make sure you're in the yolov11 directory
   cd app/yolov11
   
   # Start training
   bash train_coin.sh 1
   ```

4. **Monitor Training**:
   - Check `weights/step.csv` for training metrics
   - Best model saved as `weights/best.pt`

## Dataset Structure

The dataset has been converted from COCO format to YOLO format and organized as follows:

```
Dataset/COCO/
├── images/
│   ├── train2017/          # Training images (152 images)
│   └── val2017/            # Validation images (39 images)
├── labels/
│   ├── train2017/          # Training annotations (YOLO format)
│   └── val2017/            # Validation annotations (YOLO format)
├── train2017.txt           # Training image list
└── val2017.txt             # Validation image list
```

## Files

- `scripts/convert_coco_to_yolo.py`: Script to convert COCO annotations to YOLO format
- `main_coin.py`: Modified main training script for coin detection
- `train_coin.sh`: Training script for coin dataset
- `utils/coin_args.yaml`: Configuration file for coin detection training

## Training

### Single GPU Training
```bash
# Make sure you're in the yolov11 directory
cd app/yolov11
bash train_coin.sh 1
```

### Multi-GPU Training
```bash
# Make sure you're in the yolov11 directory
cd app/yolov11
bash train_coin.sh 2  # for 2 GPUs
```

### Manual Training
```bash
# Make sure you're in the yolov11 directory
cd app/yolov11
python3 main_coin.py --train --batch-size 16 --epochs 100
```

## Testing

```bash
# Make sure you're in the yolov11 directory
cd app/yolov11
python3 main_coin.py --test
```

## Configuration

The training configuration is in `utils/coin_args.yaml`:
- Single class: "coin" (class ID: 0)
- Batch size: 16 (reduced for smaller dataset)
- Epochs: 100 (reduced for faster training)
- Learning rate: 0.0001 to 0.01
- Augmentation: HSV, translation, scale, flip

## Dataset Statistics

- Total images: 191
- Training images: 152 (80%)
- Validation images: 39 (20%)
- Classes: 1 (coin)
- Annotations: All coins mapped to class 0

## Model Output

Trained models will be saved in the `weights/` directory:
- `best.pt`: Best model based on mAP
- `last.pt`: Last epoch model
- `step.csv`: Training metrics log
