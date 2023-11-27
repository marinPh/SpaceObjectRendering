SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi
echo "compililing constants"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/dataset_constants.py"
echo "parser"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/parser.py"
echo "importing modules"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/Utils/__init__.py"

echo "render setup"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_setup.py" "$OBJECT_NAME" "$POSE_ID"
echo "render motions"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" "$OBJECT_NAME" "$POSE_ID"