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

echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
#have an array of object names to iterate through

OBJECT_ARRAY=('6_CHEOPS_LP' '12_bepi_m' '67_p' '7_SM_MMO' '17_exoMars' ' 9_cassini_huygens' '8_xmm_newton' '14_double_star' '15_edm' '19_giotto' '13_cassini_huygens' '16_euclid' '11_bepi_mpo' '02_Hubble_Space_Telescope' '18_gaia' '21_huygens' '5_MTM' '10_cluster' '20_herschel')

for OBJECT_NAME in "${OBJECT_ARRAY[@]}"
do
  echo "Processing $object"
    
  #import the needed libraries
    blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"

  # Run the Blender script
    blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$OBJECT_NAME"
    blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py"  -- "$OBJECT_NAME" # Reads pretty_matrix.txt
done
