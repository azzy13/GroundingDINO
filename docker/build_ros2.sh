#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Building GroundingDINO ROS2 Docker Image"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Docker context: $PROJECT_ROOT"
echo ""

# Build the Docker image
docker build \
    -f "$SCRIPT_DIR/Dockerfile.ros2" \
    -t groundingdino_ros:latest \
    "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo "Image: groundingdino_ros:latest"
echo ""
echo "To test the image:"
echo "  docker run --rm --gpus all groundingdino_ros:latest ros2 node list"
echo ""
echo "To run with docker-compose:"
echo "  cd ../airsim/release_installer"
echo "  docker-compose -f docker/docker-compose.yml up groundingdino"
echo ""
