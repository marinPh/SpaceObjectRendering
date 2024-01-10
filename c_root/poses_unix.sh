#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories
SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

echo "starting imports"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"

echo "importing local modules"
 
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "./Utils/file_tools.py"


echo "create motions"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/create_motions.py" -- "$OBJECT_NAME" "$POSE_ID"

#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/parser.py"

