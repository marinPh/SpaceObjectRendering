SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"


if [ $# -ge 1 ]; then
  SEQLEN="$1"
fi

if [ $# -ge 2 ]; then
  SHIFT="$2"
fi

if [ $# -ge 3 ]; then
  ORIENTATION="$3"
fi

if [ $# -ge 4 ]; then
  ROT="$4"
fi










python3 "$SCRIPTS_DIR/crop_rot.py" --  "$SEQLEN" "$SHIFT" "$ORIENTATION" "$ROT"



# if pose_id == any other value then render the pose with the given id

