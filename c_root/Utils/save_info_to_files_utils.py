"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Saves the camera info to a file.
"""

import os
import numpy as np

import dataset_constants as dc

def save_camera_info_to_file(output_directory : str,
                             fov : int = dc.camera_fov, 
                             camera_position : np.ndarray = dc.camera_position, 
                             camera_orientation : np.ndarray = dc.camera_direction,
                             camera_info_file_name : str = dc.camera_info_file_name):
    with open(os.path.join(output_directory, camera_info_file_name), 'w') as f:
        f.write("--- Camera info ---\n")
        f.write(f"FOV: {fov}°\n")
        f.write(f"Camera position (m) [x, y, z]: [{', '.join([f'{value:.2f}' for value in camera_position])}]\n")
        f.write(f"Camera orientation (Quaternion) [w, x, y, z]): [{', '.join([f'{val:.2f}' for val in camera_orientation])}]\n")