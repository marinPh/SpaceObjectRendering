
import re
import numpy as np



def read_inertia_matrix_from_file(file_path):
    """
    Read the inertia matrix from a file.

    Args:
        file_path (str): The path to the file containing the inertia matrix.

    Returns:
        numpy.ndarray: The inertia matrix as a numpy array.

    Raises:
        ValueError: If no inertia matrix is found in the file.
    """
   
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

def extract_corners(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expressions to extract relevant information
    min_corner_pattern = re.compile(r"Min corner \(x, y, z\): \[([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\]")
    max_corner_pattern = re.compile(r"Max corner \(x, y, z\): \[([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\]")

    # Find matches using regular expressions
    min_corner_match = min_corner_pattern.search(content)
    max_corner_match = max_corner_pattern.search(content)

    if min_corner_match and max_corner_match:
        # Extract values from the matches
        min_corner_values = [float(min_corner_match.group(i)) for i in range(1, 4)]
        max_corner_values = [float(max_corner_match.group(i)) for i in range(1, 4)]

        return min_corner_values, max_corner_values
    
# Path: c_root/scripts/Utils/file_tools.py
