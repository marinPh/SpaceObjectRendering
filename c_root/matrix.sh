#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories


SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"

# Use Blender's --background option to query the Python executable
# shellcheck disable=SC2091


# List of Python modules/packages to be installed
modules=("scipy" "numpy")  # Add more as needed


# Install the modules

  # Use Blender's bundled Python to install the module

# Define default values and variables


# Parse command-line arguments
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi


blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"



# Execute your Python scripts with the full path to the Python executable
echo starting scripts
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$OBJECT_NAME"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py"  -- "$OBJECT_NAME" # Reads pretty_matrix.txt



