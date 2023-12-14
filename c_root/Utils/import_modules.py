import sys
import subprocess
import bpy

modules = ["scipy","numpy","tqdm","pyquaternion","matplotlib","pandas","open3d","pyquate","syspro"]
python_exe  = sys.executable

print(f"-----{python_exe}-----")
subprocess.run([python_exe ,"-m", "ensurepip", "--default-pip"])
for i in modules:
    subprocess.run([python_exe,"-m", "pip", "install", i])

    print(f"-----{i} installed-----")




