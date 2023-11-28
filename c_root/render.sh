SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

echo "starting imports"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/import_modules.py"

echo "compililing constants"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/dataset_constants.py"
echo "parser"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/parser.py"
echo "importing modules"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/__init__.py"


echo "render setup"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_setup.py" "$OBJECT_NAME" "$POSE_ID"

echo "render motions"
echo "be sure to have chosen the nb of samples to take as it is hard coded in render_motions.py default => 2048 or might ave been changed by accident"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" "$OBJECT_NAME" "$POSE_ID"