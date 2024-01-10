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

OBJECT_NAMES=( "27_philae" "22_SM_Integral" "6_CHEOPS" "12_bepi_m" "67_p" "28_plank" "23_iso" "36_soho" "7_SM_MMO" "17_exoMars" " 9_cassini_huygens" "8_xmm_newton" "14_double_star" "39_trace_gas_orbiter" "30_proba_3" "26_SM_MarsExpress" "32_rosetta_philae" "15_edm" "0_test_sphere" "19_giotto" "13_cassini_huygens" "9_cassini_huygens" "25_lisa_pathfinder" "24_juice_v8" "31_proba_3_ocs" "16_euclid" "35_smart_1" "11_bepi_mpo" "02_Hubble_Space_Telescope" "6_CHEOPS_LP" "42_xmm_newton" "38_tgo_edm" "40_ulysses" "34_rosetta_sc" "37_solar_orbiter" "33_schiaparelli" "18_gaia" "41_venus_express" "21_huygens" "29_SM_Proba_2" "5_MTM" "10_cluster" "20_herschel")  # Default object names


echo "OBJECT_NAMES: ${OBJECT_NAMES[@]}"
echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
echo "starting imports"

for OBJECT_NAME in "${OBJECT_NAMES[@]}"; do
  echo "Processing object: $OBJECT_NAME"
  # Add your code here to perform operations on each object
  # For example:
 # blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/calculate_inertia_matrix.py" -- "$OBJECT_NAME"
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/diagonalise_matrix.py" -- "$OBJECT_NAME"
done

echo "imports done"

# Execute your Python scripts with the full path to the Python executable



