#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories


SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"

# Use Blender's --background option to query the Python executable
# shellcheck disable=SC2091


# List of Python modules/packages to be installed


# Install the modules

  # Use Blender's bundled Python to install the module

# Define default values and variables


# Parse command-line arguments
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi


if [ "$OBJECT_NAME" == "-h" ] || [ "$OBJECT_NAME" == "--help" ]; then
  echo "Usage: ./matrix.sh [object_name]"
  
  echo "object_name: name of the object to calculate"
  echo "if first argument == -a or --all then generate matrices for all objects from 000 to number following"
  echo "if first argument == -h or --help then print help"
  exit 0
  fi
echo "OBJECT_NAME: $OBJECT_NAME"
echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
echo "starting imports"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "./Utils/import_modules.py"
echo "imports done"


if [ "$OBJECT_NAME" == "-a" ] || [ "$OBJECT_NAME" == "--all" ]; then
#use ./Utils/scrap_object_names.py to get the list of all the objects
  list_names=$(python3 "./Utils/scrap_object_names.py")
  echo "list_names: $list_names"
  for name in $list_names;
  do
    echo "Processing object: $name"
    # Add your code here to perform operations on each object
    # For example:
   
    blender "$OBJECT_DIR/$name.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$name"
    blender "$OBJECT_DIR/$name.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py" -- "$name"
  done
  exit 0
fi

# Execute your Python scripts with the full path to the Python executable

echo "starting matrix calculation"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$OBJECT_NAME"
echo "matrix calculation done"
echo "starting diagonalisation"
echo "--mainobj = ${OBJECT_NAME}"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py"  -- "$OBJECT_NAME" # Reads pretty_matrix.txt
echo "diagonalisation done"
echo "next .sh to use is pose.sh"


