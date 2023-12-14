import os
import sys
import re


# Create parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
blend_file_path = os.path.join(parent_dir,"c_root", "objects", "blend")
#look for all files that end with .blend
pattern = re.compile(r".blend$")
# Create list of all blend files withouth the .blend extension
blend_files = [f.split(".")[0] for f in os.listdir(blend_file_path) if pattern.search(f)]
# create a string of all blend files in that can be copied into a .sh file
blend_files_string = "\""
for file in blend_files:
    blend_files_string += file + "\" \""
print(blend_files_string+"\"")
