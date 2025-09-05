"""
YOLOv11 Detector
use yolov11 based algorithms for circle detection
"""

import os
import sys
import cv2
import numpy as np
import torch
import yaml

# Add the yolov11 directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .nets import nn
from .utils import util

# Create module mapping for loading models trained with original module structure
import importlib.util
import types

# Create a temporary module for 'nets' to handle model loading
nets_module = types.ModuleType('nets')
nets_module.nn = sys.modules['features.yolov11.nets.nn']
sys.modules['nets'] = nets_module

# Create a temporary module for 'utils' to handle model loading  
utils_module = types.ModuleType('utils')
utils_module.util = sys.modules['features.yolov11.utils.util']
sys.modules['utils'] = utils_module

# Also create the nested module structure that the model expects
sys.modules['nets.nn'] = sys.modules['features.yolov11.nets.nn']
sys.modules['utils.util'] = sys.modules['features.yolov11.utils.util']


class YOLOv11Detector:
    """
    YOLOv11-based object detector for circular objects.
    
    This class provides a clean interface to the YOLOv11 model
    with minimal changes to the original implementation.
    """
    
    def __init__(self, model_path="features/yolov11/weights/best.pt", input_size=640):
        """
        Initialize the YOLOv11 detector.
        
        Args:
            model_path: Path to the trained YOLOv11 model
            input_size: Input size for YOLOv11 model
        """
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLOv11 model
        self.model = self._load_model(model_path)
        
        # Load class names
        self.class_names = self._load_class_names()
        
    def _load_model(self, model_path):
        """Load the trained YOLOv11 model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading YOLOv11 model from: {model_path}")
        model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        model = model_data['model'].float().fuse()
        model.half()  # Use half precision for faster inference
        model.eval()
        model.to(self.device)
        
        return model
    
    def _load_class_names(self):
        """Load class names from coin_args.yaml."""
        yaml_path = "features/yolov11/utils/coin_args.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('names', {0: 'coin'})
        return {0: 'coin'}
    
    def _preprocess_image(self, image):
        """
        Preprocess image for YOLOv11 inference.
        
        Returns:
            preprocessed_tensor: Preprocessed image tensor
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
        """
        h, w = image.shape[:2]
        
        # Resize image to model input size while maintaining aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.half()  # Convert to half precision to match model
        
        return tensor.to(self.device), scale, scale
    
    def _postprocess_detections(self, outputs, scale_x, scale_y, original_shape):
        """
        Postprocess YOLOv11 outputs to get bounding boxes in original image coordinates.
        
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # Apply NMS
        outputs = util.non_max_suppression(outputs, confidence_threshold=0.25, iou_threshold=0.45)
        
        detections = []
        if outputs[0] is not None and len(outputs[0]) > 0:
            for detection in outputs[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                
                # Scale back to original image coordinates
                x1 = int(x1 / scale_x)
                y1 = int(y1 / scale_y)
                x2 = int(x2 / scale_x)
                y2 = int(y2 / scale_y)
                
                # Clamp to image boundaries
                h, w = original_shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def detect(self, image):
        """
        Detect objects using YOLOv11 model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples representing detected objects
        """
        # Preprocess image
        tensor, scale_x, scale_y = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(tensor)
        
        # Postprocess detections
        detections = self._postprocess_detections(outputs, scale_x, scale_y, image.shape)
        
        return detections
    
    def get_class_name(self, class_id):
        """Get class name for a given class ID."""
        return self.class_names.get(class_id, f"class_{class_id}")
