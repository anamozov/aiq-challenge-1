# AIQ Challenge 1 - Frontend Interface

A comprehensive web interface for the Coin Detection API with full endpoint coverage.

## Quick Start

### Option 1: Using the Python Server
```bash
# Start the frontend server
python3 serve_frontend.py

# Access the frontend
# Local: http://localhost:8080/frontend.html
# Remote: http://10.227.228.64:8080/frontend.html
```

### Option 2: Direct File Access
Simply open `frontend.html` in your web browser.

## Features

### API Status & Health
- Real-time API health monitoring
- Direct link to API documentation

### Image Management
- Upload images for detection
- List all uploaded images
- Get specific image details
- View image metadata

### Detection & Analysis
- Run coin detection on images
- Get objects for specific images
- View detection results and statistics

### Object Management
- List all detected objects
- Get specific object details
- Filter by image ID
- Delete all objects

### Visualization Tools
- Combined: Shows detection boxes and mask side by side
- Detection: Shows only YOLO bounding boxes
- Mask: Shows only circular mask
- Download generated visualizations

### Batch Operations
- Batch detection on multiple images
- Batch visualization generation
- Process multiple images at once

### Advanced Features
- Export all data to JSON
- Clear all data
- System information
- Quick system test
- Statistics and analytics

## Usage

1. **Upload an Image**: Select an image file and click "Upload Image"
2. **Run Detection**: Enter the image ID and click "Run Detection"
3. **Generate Visualization**: Choose visualization type and generate
4. **Download Results**: Click "Download Image" to save visualizations
5. **Batch Operations**: Use comma-separated image IDs for batch processing

## Configuration

The frontend is pre-configured to work with the API at `http://10.227.228.64:8000`. To change the API address, edit the `API_BASE` variable in the HTML file.

## Layout

The interface uses a two-column responsive layout:
- **Left Column**: Core operations (API status, image management, detection, objects)
- **Right Column**: Advanced features (visualization, batch operations, advanced features)

## Responsive Design

The interface works on:
- Desktop computers
- Tablets
- Mobile phones

## Security Note

This is a development frontend. For production use, consider:
- Adding authentication
- Implementing proper error handling
- Adding input validation
- Using HTTPS
