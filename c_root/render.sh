SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/dataset_constants.py"
blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_setup.py" "$OBJECT_NAME" "$POSE_ID"

eval "blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P $SCRIPTS_DIR/render_motions.py" "$OBJECT_NAME" "$POSE_ID"