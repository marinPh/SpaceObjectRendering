import os
import sys

#path to objects/blender/
#get from command line motion_id
#input_name = sys.argv[sys.argv.index("--") + 1]
input_name = "6_1"
blender_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"objects","blend")

object_id = input_name.split("_")[0]


#find the name of the object
for f in os.listdir(blender_path):
    if f.endswith(".blend"):
        if f.startswith(f"{object_id}_"):
            object_name = f[:-6]
            break
print (object_name)