import os
import sys
import re


# Create parent directory path
def scrap_names():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    blend_file_path = os.path.join(parent_dir, "objects", "blend")
    #look for all files that end with .blend
    pattern = re.compile(r".blend$")
    # Create list of all blend files without .blend extension
    blend_files = [file[:-6] for file in os.listdir(blend_file_path) if pattern.search(file)]
    
    
    
    
    # create a string of all blend files in that can be copied into a .sh file
    blend_files_string = ""
    for file in blend_files:
        blend_files_string += file +" "
   

    return blend_files_string

print(scrap_names())


