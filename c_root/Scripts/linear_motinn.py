import numpy as np
import os
import sys

# User-defined inputs
# name of the object
obj_name = "6_CHEOPS_LP"
obj_id = obj_name.split("_")[0]
output_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input", f"{obj_id}_{99}")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
min_distance =4.145760260795728
#for each iteration write a line in the file
for i in range(100):
    st =f" {i},6,0,0,0,0,0,0,{min_distance+i}"
    open(os.path.join(output_directory,"scene_gt.txt"), "a").write(st+"\n")
    
    