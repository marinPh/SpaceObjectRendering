import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python calculate_inertia_matrix.py <arg1>")
    sys.exit(1)

# Access the argument passed in the command line
arg1 = sys.argv[1]
print("Argument 1:", arg1)
module = arg1.split("/")[-1].split()
python_exe = sys.executable

subprocess.run([python_exe,"-m", "pip", "install", module])
print(f"-----{module} installed-----")

