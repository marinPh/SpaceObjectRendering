import sys
import subprocess
import bpy

modules = ["scipy","numpy"]
python_exe = subprocess.run(["which", "blender"], text=True, capture_output=True).stdout.strip()
print(f"{python_exe} is exec dir")

print(sys.path)
for i in modules:
    subprocess.check_call([python_exe,"-m", "pip", "install", i,"-t",i])
    print(f"{i} installed")

