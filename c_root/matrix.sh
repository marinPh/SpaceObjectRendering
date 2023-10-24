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
echo "OBJECT_NAME: $OBJECT_NAME"
echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
echo "starting imports"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"
echo "imports done"



# Execute your Python scripts with the full path to the Python executable

echo "starting matrix calculation"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$OBJECT_NAME"
echo "matrix calculation done"
echo "starting diagonalisation"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py"  -- "$OBJECT_NAME" # Reads pretty_matrix.txt
echo "diagonalisation done"
echo "next .sh to use is pose.sh"


