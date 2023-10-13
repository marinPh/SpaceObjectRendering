SCRIPTS_DIR="./scripts"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

blender -b -P "$SCRIPTS_DIR/render_setup.py" "$OBJECT_NAME" "$POSE_ID"



eval "blender -b -P $SCRIPTS_DIR/render_motions.py" "$OBJECT_NAME" "$POSE_ID"