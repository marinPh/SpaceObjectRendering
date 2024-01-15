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

if [ "$OBJECT_NAME" == "-h" ] || [ "$OBJECT_NAME" == "--help" ]; then
  echo "Usage: ./render.sh [object_name] [pose_id]"
  echo "if POSE_ID == 000 then generate random poses"
  echo "object_name: name of the object to motion"
  echo "pose_id: id of the pose to generate"
  echo "if first argument == -a or --all then generate poses for all objects from 000 to number following"
  echo "if first argument == -h or --help then print help"
  exit 0
  fi


  echo "OBJECT_NAME: $OBJECT_NAME"
  echo "POSE_ID: $POSE_ID"



if [ "$OBJECT_NAME" == "-a" ] || [ "$OBJECT_NAME" == "--all" ]; then
#use ./Utils/scrap_object_names.py to get the list of all the objects
  list_names=$(python3 "./Utils/scrap_object_names.py")
  echo "list_names: $list_names"
  for name in $list_names;
  do
    echo "Processing object: $name"
    # Add your code here to perform operations on each object
    # For example:
   
    blender "$OBJECT_DIR/$name.blend" -b -P "$SCRIPTS_DIR/create_random_poses.py" -- "$name"
    #for in range(pose_id)
    echo "pose_id: $POSE_ID"
    for ((i = 1; i <= POSE_ID; i++)); do
      blender "$OBJECT_DIR/$name.blend" -b -P "$SCRIPTS_DIR/create_motions.py" -- "$name" "$i"
    done
  done
  exit 0
fi

echo "starting imports"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"

echo "importing local modules"
 
if [ "$POSE_ID" == "000" ]; then
  echo "generate random poses"
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/create_random_poses.py" -- "$OBJECT_NAME"
  exit 0
fi


echo "create motions"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/create_motions.py" -- "$OBJECT_NAME" "$POSE_ID"

#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/parser.py"

