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




# Execute your Python scripts with the full path to the Python executable

echo "starting matrix calculation"
blender "$OBJECT_DIR/0_test_sphere.blend" -b -P "./test_ray.py"


