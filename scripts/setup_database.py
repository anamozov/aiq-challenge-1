#!/usr/bin/env python3
"""
Database setup script
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.database import create_tables
from core.config import settings

def setup_database():
    """
    Create all database tables
    """
    print("Setting up database...")
    print(f"Database URL: {settings.database_url}")
    print(f"Storage directory: {settings.storage_dir}")
    
    # Create tables
    create_tables()
    
    print("Database setup completed successfully!")
    print("Tables created:")
    print("- images")
    print("- detections") 
    print("- evaluations")

if __name__ == "__main__":
    setup_database()
