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


 OBJECT_NAMES=( "9_cassini_huygens" "02_Hubble_Space_Telescope" "0_test_sphere" "10_cluster" "11_bepi_mpo" "12_bepi_m" "13_cassini_huygens" "14_double_star" "15_edm" "16_euclid" "17_exoMars" "18_gaia" "19_giotto" "20_herschel" "21_huygens" "22_SM_Integral" "23_iso" "24_juice_v8" "5_MTM" "67_p" "6_CHEOPS_LP" "7_SM_MMO" "8_xmm_newton")  # Default object names


echo "OBJECT_NAMES: ${OBJECT_NAMES[@]}"
echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
echo "starting imports"

for OBJECT_NAME in "${OBJECT_NAMES[@]}"; do
  echo "Processing object: $OBJECT_NAME"
  # Add your code here to perform operations on each object
  # For example:
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "./Utils/tree_viewer_editor.py" -- "$OBJECT_NAME"
done