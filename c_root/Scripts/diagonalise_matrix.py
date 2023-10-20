"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: This script reads in the inertia matrix from a text file and diagonalizes it using two methods:
                1) Zeroing out all non-diagonal components
                2) Calculating the eigenvectors and eigenvalues
"""
import argparse

import numpy as np
import os
from scipy.spatial.transform import Rotation
import re

################################################
# User-defined inputs

# Input and output directory for the inertia matrix text file
proj_dir : str = os.path.dirname(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('object_name', help='Name of the object')
args = parser.parse_args()

main_obj_name = args.object_name
input_output_directory : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")
# Input file name (should correspond to the output file name of the inertia matrix calculator)
input_file_name = 'inertia_matrix.txt'
# Name of the output text files
# The object name will be prepended to the output file name
output_file_name = 'diagonalized_inertia_matrix.txt'

################################################
def write_pretty_file(matrix):
    with open(output_dir,"w") as file:
        formatted_matrix = ",\n".join(["    [{:6.2f}, {:6.2f}, {:6.2f}]".format(*row) for row in matrix])
        file.write("np.array([\n" + formatted_matrix + "\n])")
def zero_non_diagonal(matrix):
    diag_matrix = np.zeros_like(matrix)
    np.fill_diagonal(diag_matrix, np.diagonal(matrix))
    return diag_matrix

def diagonalize_with_eigenvectors(matrix):
    # Calculate the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Diagonalize the inertia matrix using the eigenvectors
    diagonal_matrix = np.diag(eigenvalues)

    return diagonal_matrix, eigenvectors

def rotation_matrix_to_euler_angles(matrix):
    rot = Rotation.from_matrix(matrix)
    return rot.as_euler('xyz', degrees=True)

def calculate_inaccuracy(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)

def read_inertia_matrix_from_file(file_path):
    with open(file_path, 'r') as f:
        # Read the entire file content
        file_content = f.read()

        # Find the numpy array string using a regular expression
        if res := re.search(r'np.array\(\[([\s\S]*?)\]\)', file_content):
            matrix_string = res[1]
        else:
            raise ValueError("No inertia matrix found in file.")

        # Remove square brackets and spaces, then convert the matrix string into a list of lists of floats
        matrix = np.array([list(map(float, re.sub(r'[\[\]\s]', '', line).split(','))) for line in matrix_string.strip().split(',\n') if line.strip()])

    return matrix

def extract_object_name(file_content):
    if match := re.search(r"object '(.+?)'", file_content):
        return match[1]
    else:
        return "Unnamed Object"

def print_matrix(matrix):
    formatted_matrix = ",\n".join(["    [{:6.2f}, {:6.2f}, {:6.2f}]".format(*row) for row in matrix])
    return "np.array([\n" + formatted_matrix + "\n])"

def write_inaccuracy_and_rotation_to_file(file, method_name, inaccuracy, rotation_degrees=None):
    file.write("{} Inaccuracy: {:6.2f}\n".format(method_name, inaccuracy))
    if rotation_degrees is not None:
        file.write("{} Rotation applied (in degrees): [{:6.2f}, {:6.2f}, {:6.2f}]\n".format(method_name, *rotation_degrees))
    file.write("\n")

# Load the inertia matrix from the file
input_file_path = os.path.join(input_output_directory, input_file_name)
with open(input_file_path, 'r') as f:
    file_content = f.read()
object_name = main_obj_name
inertia_matrix = read_inertia_matrix_from_file(input_file_path)

# Calculate the diagonal matrices
diag_matrix_zero = zero_non_diagonal(inertia_matrix)
diag_matrix_eig, eigenvectors = diagonalize_with_eigenvectors(inertia_matrix)

# Calculate the rotation in degrees
rotation_degrees = rotation_matrix_to_euler_angles(eigenvectors)

# Calculate inaccuracy for both methods
inaccuracy_zero = calculate_inaccuracy(inertia_matrix, diag_matrix_zero)
inaccuracy_eig = calculate_inaccuracy(inertia_matrix, diag_matrix_eig)
#
write_pretty_file(diag_matrix_eig)
# Print the results
print(f"Original inertia matrix:\n{print_matrix(inertia_matrix)}\n")
print(f"Diagonal matrix with non-diagonal components zeroed out:\n{print_matrix(diag_matrix_zero)}\n")
print("Inaccuracy with non-diagonal components zeroed out: {:6.2f}\n".format(inaccuracy_zero))
print(f"Diagonal matrix after calculating eigenvectors:\n{print_matrix(diag_matrix_eig)}\n")
print(
    f"Eigenvectors used for diagonalization:\n{print_matrix(eigenvectors)}\n"
)
print("Rotation applied using eigenvectors (in degrees): [{:6.2f}, {:6.2f}, {:6.2f}]\n".format(*rotation_degrees))
print("Inaccuracy with diagonalization using eigenvectors: {:6.2f}\n".format(inaccuracy_eig))

# Write the results to the output file
output_file_name_with_object = f"{object_name}_{output_file_name}"
output_file_path = os.path.join(input_output_directory, output_file_name_with_object)
with open(output_file_path, 'w') as output_file:
    output_file.write(f"Original inertia matrix for object '{object_name}':\n")
    output_file.write(print_matrix(inertia_matrix) + "\n\n")

    output_file.write(
        f"Total combined inertia matrix for object '{object_name}' (zeroing out all non-diagonal components):\n"
    )
    output_file.write(print_matrix(diag_matrix_zero) + "\n\n")
    write_inaccuracy_and_rotation_to_file(output_file, "Zeroing out non-diagonal components", inaccuracy_zero)

    output_file.write(
        f"Total combined inertia matrix for object '{object_name}' (using eigenvectors):\n"
    )
    output_file.write(print_matrix(diag_matrix_eig) + "\n\n")
    write_inaccuracy_and_rotation_to_file(output_file, "Diagonalization using eigenvectors", inaccuracy_eig, rotation_degrees)
    
    output_file.write(
        f"The inaccuracy is the Frobenius norm (like a Euclidean distance) between the original matrix and the matrix after diagonalization.\n"
    )



