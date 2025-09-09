#!/bin/bash

# Docker Testing Script for AIQ Challenge 1

echo "🐳 AIQ Challenge 1 - Docker Testing Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists docker; then
    print_error "Docker is not installed!"
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed!"
    exit 1
fi

print_success "Prerequisites check passed"

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose down >/dev/null 2>&1

# Build and start services
print_status "Building and starting services..."
if docker-compose up -d --build; then
    print_success "Services started successfully"
else
    print_error "Failed to start services"
    exit 1
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Test API health
print_status "Testing API health..."
if curl -s http://localhost:8000/health >/dev/null; then
    print_success "API is healthy"
else
    print_warning "API health check failed, but continuing..."
fi

# Test frontend
print_status "Testing frontend..."
if curl -s http://localhost:8080/ >/dev/null; then
    print_success "Frontend is accessible"
else
    print_warning "Frontend check failed, but continuing..."
fi

# Show service status
print_status "Service status:"
docker-compose ps

# Show access URLs
echo ""
print_status "Access URLs:"
echo "  Frontend: http://10.227.228.64:8080/frontend.html"
echo "  API: http://10.227.228.64:8000"
echo "  API Docs: http://10.227.228.64:8000/docs"
echo ""

# Show logs
print_status "Recent logs:"
docker-compose logs --tail=20

echo ""
print_success "Docker testing setup complete!"
print_status "You can now test the application using the URLs above"
print_status "To stop the services, run: docker-compose down"
