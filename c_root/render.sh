SCRIPTS_DIR="./scripts"
OBJECT_DIR="./objects/blend"
if [ $# -ge 1 ]; then
  OBJECT_NAME="$1"
fi

if [ $# -ge 2 ]; then
  POSE_ID="$2"
fi

if [ $# -ge 3 ]; then
  FLAG="$3"
fi

#if object name == -h or --help then print help
if [ "$OBJECT_NAME" == "-h" ] || [ "$OBJECT_NAME" == "--help" ]; then
  echo "Usage: ./render.sh [object_name] [pose_id] [flag]"
  echo "object_name: name of the object to render"
  echo "pose_id: id of the pose to render"
  echo "flag: if set to 1, will render the object with the flag"
  echo "if second argument == -a or --all then render all poses in input"
  echo "if second argument == -h or --help then print help"
  echo " if last argument == -b adds background to all rendered images"
  exit 0
fi




echo "OBJECT_NAME: $OBJECT_NAME"
echo "POSE_ID: $POSE_ID"
echo "FLAG: $FLAG"

#if -a or --all then render all poses, find all the find all the poses in the poses folder and render them
if [ "$OBJECT_NAME" == "-a" ] || [ "$OBJECT_NAME" == "--all" ]; then
  echo "rendering all poses"
  #find all the poses in the poses folder and render them
  for f in "./input/"*; do
    # if f == ./input/earthImg/ or ./input/backimg then skip
    if [ "$f" == "./input/earthImg" ] || [ "$f" == "./input/backimg" ]; then
      continue
    fi
    pose=$(echo "$f" | cut -d'/' -f3)
    echo "rendering pose: $f"
    echo "pose: $pose"

    #use the script fromFileToName.py to get the name of the pose from its id
    NAME=$(python3 "./Utils/fromFileToName.py" "$pose")
    echo "NAME: $NAME"

    #split the pose id to get the second part
    sp=$(echo "$pose" | cut -d'_' -f2)
    echo "sp: $sp"
    
    blender "$OBJECT_DIR/$NAME.blend" -b -P "$SCRIPTS_DIR/render_setup.py" "$NAME" "$sp"
    blender "$OBJECT_DIR/$NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" "$NAME" "$sp"
    # if flag == -b then add background to all rendered images
    if [ "$FLAG" == "-b" ]; then
      echo "adding background"
      
      # if f.split(_)[1] == 000 then use BackProcRdm.py
        # else use BackProcSeq.py
        if [ "$second_part" == "000" ]; then
            echo "using BackProcRdm.py"
            blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/BackProcRdm.py" -- "$NAME" "$sp" "$FLAG"
            else
            echo "using BackProcSeq.py"
            blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/BackProcSeq.py" -- "$NAME" "$sp" "$FLAG"
            fi
     
    fi
  done
  exit 0
fi

# if pose_id == any other value then render the pose with the given id

echo "render setup"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_setup.py" "$OBJECT_NAME" "$POSE_ID"
#blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/render_motions.py" "$OBJECT_NAME" "$POSE_ID"

# if flag == -b then add background to all rendered images
echo "FLAG: $FLAG"
if [ "$FLAG" == "-b" ]; then
  echo "adding background"
  blender "$OBJECT_DIR/$OBJECT_NAME.blend" -b -P "$SCRIPTS_DIR/BackProcSeq.py" -- "$OBJECT_NAME" "$POSE_ID" "$FLAG"
fi