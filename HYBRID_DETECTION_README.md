# Hybrid Detection Algorithm

This hybrid detection algorithm which merges YOLOv11 based coin detection model and cv-based coin foreground extraction algorithm 

## Prerequisites

1. **Trained YOLOv11 Model**: The system requires a trained YOLOv11 model at `weights/best.pt`
2. **Dependencies**: All dependencies from `requirements.txt` must be installed
3. **CUDA**: GPU acceleration for yolov11

## Usage

### Basic Usage

```bash
python3 hybrid_detection.py <image_path>
```

### Advanced Usage

```bash
python3 hybrid_detection.py <image_path> --model-path weights/best.pt --output-dir results/ --input-size 640
```

### Parameters

- `image_path`: Path to the input image
- `--model-path`: Path to the YOLOv11 model (default: `weights/best.pt`)
- `--output-dir`: Output directory for results (default: `results/`)
- `--input-size`: YOLOv11 input size (default: 640)


## How It Works

### Step 1: YOLOv11 Detection
- Loads the trained YOLOv11 model
- Preprocesses the input image to the model's input size
- Runs inference to detect potential coin regions
- Applies non-maximum suppression to filter detections

### Step 2: Coin Foreground Extraction
- Crops each detected region with padding
- Applies the `detect.py` algorithm to each cropped region
- Uses both contour-based and Hough circle detection for robust coin detection
- Transforms detected circles back to original image coordinates

### Step 3: Mask Creation
- Creates a binary mask in the original image coordinates
- Draws circles for all detected coins
- Generates visualization with original image and mask side by side
