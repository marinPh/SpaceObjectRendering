#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories
SCRIPTS_DIR="./scripts"

# Define default values and variables


# Parse command-line arguments
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

# ... Rest of the script ...



# Execute your Python scripts with the full path to the Python executable
blender -b -P "$SCRIPTS_DIR/tumble_function.py" "$OBJECT_NAME" # Reads pretty_matrix.txt
blender -b -P "$SCRIPTS_DIR/create_poses.py" "$OBJECT_NAME" "$POSE_ID"
blender -b -P "$SCRIPTS_DIR/pose_to_motion.py" "$OBJECT_NAME" "$POSE_ID"
blender -b -P "$SCRIPTS_DIR/create_multi_poses.py" "$OBJECT_NAME" "$POSE_ID"
blender -b -P "$SCRIPTS_DIR/multi_pose_to_motions.py" "$OBJECT_NAME" "$POSE_ID"




# ... Rest of the script ...
