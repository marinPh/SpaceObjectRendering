import os
import re
import sys  

#path to objects/blender/
blender_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"objects","blend")
#put in an array all the file names without .blend extension
files = [f[:-6] for f in os.listdir(blender_path) if f.endswith(".blend")]

print (files)