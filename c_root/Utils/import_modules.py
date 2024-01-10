import sys
import subprocess
import bpy
import site

# Get the user-specific site-packages directory
user_site_packages = site.getusersitepackages()
print (user_site_packages)
sys.path.append(user_site_packages)
modules = ["scipy","numpy","tqdm","pyquaternion","matplotlib","pandas","open3d","pyquate","syspro","pymeshlab"]
python_exe  = sys.executable

print(f"-----{python_exe}-----")

subprocess.run([python_exe,"-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([python_exe ,"-m", "ensurepip", "--default-pip"])

for i in modules:
    subprocess.run([python_exe,"-m", "pip", "install", i])

    print(f"-----{i} installed-----")

subprocess.run([python_exe,"-c", "from distutils.sysconfig import get_python_lib; print(get_python_lib())"])

print(sys.path)




