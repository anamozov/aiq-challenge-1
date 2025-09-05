#!/usr/bin/env python3
"""
Simple test script to verify the API setup
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test core modules
        from core.config import settings
        print("✓ Core config imported")
        
        from core.storage import save_uploaded_file
        print("✓ Core storage imported")
        
        # Test models
        from models.database import get_db, create_tables
        print("✓ Models database imported")
        
        from models.tables import Image, Detection, Evaluation
        print("✓ Models tables imported")
        
        from models.schemas import Image as ImageSchema, Detection as DetectionSchema
        print("✓ Models schemas imported")
        
        # Test services
        from services.detection_service import DetectionService
        print("✓ Detection service imported")
        
        # Test API endpoints
        from api.endpoints import images, objects
        print("✓ API endpoints imported")
        
        # Test features
        from features.yolov11 import YOLOv11Detector
        print("✓ YOLOv11 detector imported")
        
        from features.cv_detection import CVDetector
        print("✓ CV detector imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_setup():
    """Test database setup"""
    try:
        print("\nTesting database setup...")
        
        from models.database import create_tables
        from core.config import settings
        
        print(f"Database URL: {settings.database_url}")
        print(f"Storage directory: {settings.storage_dir}")
        
        # Create tables
        create_tables()
        print("✓ Database tables created")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def test_detection_service():
    """Test detection service initialization"""
    try:
        print("\nTesting detection service...")
        
        from services.detection_service import DetectionService
        
        # Initialize service (this will test if YOLO model exists)
        service = DetectionService()
        print("✓ Detection service initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection service error: {e}")
        print("Note: This is expected if YOLO model is not trained yet")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("API Setup Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test database setup
    success &= test_database_setup()
    
    # Test detection service (optional)
    test_detection_service()  # Don't fail if YOLO model not available
    
    print("\n" + "=" * 50)
    if success:
        print("✅ API setup test completed successfully!")
        print("You can now run: python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)
