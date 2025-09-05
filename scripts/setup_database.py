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
from models.tables import Image, Detection, Evaluation  # Import models to register them
from core.config import settings

def setup_database():
    """
    Create all database tables
    """
    print("Setting up database...")
    print(f"Database URL: {settings.database_url}")
    print(f"Storage directory: {settings.storage_dir}")
    
    # Create tables using raw SQL (more reliable than SQLAlchemy create_all)
    import sqlite3
    conn = sqlite3.connect('storage/detection.db')
    cursor = conn.cursor()
    
    # Create images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            filename VARCHAR NOT NULL,
            original_filename VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            object_id VARCHAR NOT NULL UNIQUE,
            bbox_x1 FLOAT NOT NULL,
            bbox_y1 FLOAT NOT NULL,
            bbox_x2 FLOAT NOT NULL,
            bbox_y2 FLOAT NOT NULL,
            centroid_x FLOAT NOT NULL,
            centroid_y FLOAT NOT NULL,
            radius FLOAT NOT NULL,
            confidence FLOAT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
    ''')
    
    # Create evaluations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            evaluation_type VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value FLOAT NOT NULL,
            details TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database setup completed successfully!")
    print("Tables created:")
    print("- images")
    print("- detections") 
    print("- evaluations")

if __name__ == "__main__":
    setup_database()
