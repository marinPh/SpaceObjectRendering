#!/bin/bash

# Define paths to your Blender executable, object, and matrix directories


SCRIPTS_DIR="./scripts"

# Determine the Python executable using Python itself
python_exe= blender

# List of Python modules/packages to be installed
modules=("scipy" "numpy")  # Add more as needed

# Install the modules
for module in "${modules[@]}"; do
    $python_exe -m pip install "$module"
done

# Define default values and variables


# Parse command-line arguments
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi


# ... Rest of the script ...

# Execute your Python scripts with the full path to the Python executable
blender -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" "$OBJECT_NAME"
blender -b -P "$SCRIPTS_DIR/diagonalise_matrix.py" "$OBJECT_NAME" # Reads pretty_matrix.txt

# Render the scenes with Blender

# ... Rest of the script ...
