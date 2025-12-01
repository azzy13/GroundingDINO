#!/bin/bash
set -e

echo "========================================"
echo "GroundingDINO ROS2 Container Starting"
echo "========================================"

# Source ROS2 Humble
echo "Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f "$ROS2_WS/install/setup.bash" ]; then
    echo "Sourcing ROS2 workspace..."
    source $ROS2_WS/install/setup.bash
else
    echo "Warning: ROS2 workspace not built yet"
fi

# Add GroundingDINO to Python path
export PYTHONPATH=$GROUNDINGDINO_PATH:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Print environment info
echo "----------------------------------------"
echo "Environment Information:"
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "GROUNDINGDINO_PATH: $GROUNDINGDINO_PATH"
echo "ROS2_WS: $ROS2_WS"
echo "----------------------------------------"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Devices:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "----------------------------------------"
else
    echo "Warning: CUDA not available"
    echo "----------------------------------------"
fi

echo "Starting GroundingDINO node..."
echo "========================================"
echo ""

# Execute command
exec "$@"
