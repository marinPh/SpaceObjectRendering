SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"



echo "starting imports"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"

echo "importing local modules"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"
echo "loop through poses"






 OBJECT_NAMES=("02_Hubble_Space_Telescope" "10_cluster" "11_bepi_mpo" "12_bepi_m" "13_cassini_huygens" "14_double_star" "15_edm" "16_euclid" "17_exoMars" "18_gaia" "19_giotto" "20_herschel" "21_huygens" "22_SM_Integral" "23_iso" "24_juice_v8" "5_MTM" "67_p" "6_CHEOPS_LP" "7_SM_MMO" "8_xmm_newton")  # Default object names


echo "OBJECT_NAMES: ${OBJECT_NAMES[@]}"
echo "OBJECT_DIR: $OBJECT_DIR"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
echo "starting imports"

for OBJECT_NAME in "${OBJECT_NAMES[@]}"; do
  echo "Processing object: $OBJECT_NAME" 
  # Add your code here to perform operations on each object

  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" -- "$OBJECT_NAME" "1"

  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" -- "$OBJECT_NAME" "2"

  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" -- "$OBJECT_NAME" "000"

done
