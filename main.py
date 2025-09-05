#!/usr/bin/env python3
"""
FastAPI Application for AIQ Challenge 1
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import images, objects
from core.config import settings

# Create FastAPI app
app = FastAPI(
    title="AIQ Challenge 1 API",
    description="API for detecting and managing circular objects in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])
app.include_router(objects.router, prefix="/api/v1/objects", tags=["objects"])

@app.get("/")
def read_root():
    return {"message": "AIQ Challenge 1 API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
