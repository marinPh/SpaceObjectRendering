#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories
OBJECT_DIR="./objects/blend"
SCRIPTS_DIR="./scripts"

# Define default values and variables


# Parse command-line arguments
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi
echo "---OBJECT_NAME: $OBJECT_NAME"
echo "---OBJECT_DIR: $OBJECT_DIR"
echo "---SCRIPTS_DIR: $SCRIPTS_DIR"

# ... Rest of the script ...
echo "rechecking imports"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/apply_object_id_to_satellite_part.py"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/create_sun.py"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/dataset_constants.py"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/save_info_to_files_utils.py"

blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/__init__.py"
# Execute your Python scripts with the full path to the Python executable
echo "---calling tumble function"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/tumble_motion_function.py" --python "$OBJECT_NAME" "$POSE_ID"
echo "---calling create poses"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/create_poses.py" "$OBJECT_NAME" "$POSE_ID"
echo "---calling pose to motion"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/pose_to_motion.py" "$OBJECT_NAME" "$POSE_ID"
echo "---calling create multi poses"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/create_multi_poses.py" "$OBJECT_NAME" "$POSE_ID"
echo "---calling multi pose to motions"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/multi_pose_to_motions.py" "$OBJECT_NAME" "$POSE_ID"



# ... Rest of the script ...
