"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Generates random poses for a single object.
"""


import numpy as np
import os
from pyquaternion import Quaternion

import sysprj

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.save_info_to_files_utils import save_camera_info_to_file
import Utils.dataset_constants as dc

################################################
# User-defined inputs
proj_dir : str = os.path.dirname(os.path.dirname(__file__))
input_dir : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")




# Number of poses to generate
num_poses : int = dc.val_num_poses

# ID of the object to be rendered
object_id : str = "04"

# Whether the sun orientation is randomly generated or not
sun_rnd_generated : bool = True

################################################
# Calculated constants

output_directory = (
    os.path.join(output_directory, object_id + dc.random_poses_motion_id)
)
print(output_directory)

################################################

def main():
    generate_random_poses(num_poses, object_id, output_directory, sun_rnd_generated)


def generate_random_poses(num_poses : int, object_id : str,
                            output_directory : str,
                            sun_rnd_generated : bool,
                            num_frustums : int = dc.nb_layers, 
                            min_distance : float = dc.min_distance, 
                            max_distance : float = dc.max_distance,
                            fov : int = dc.camera_fov, 
                            camera_position : np.ndarray = dc.camera_position, 
                            camera_direction : np.ndarray = dc.camera_direction,
                            scene_info_file_name : str = dc.scene_info_file_name, 
                            sun_orientations_file_name : str = dc.sun_orientations_file_name,
                            scene_gt_file_name : str = dc.scene_gt_file_name,
                            camera_info_file_name : str = dc.camera_info_file_name):

    positions, rotations = generate_non_uniform_poses(num_frustums, num_poses, max_distance, min_distance, fov, camera_position)

    # Create the output directory if it doesn't exist yet
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if sun_rnd_generated:
        sun_orientations = generate_random_sun_orientations(num_poses)
        sun_orientations = np.array(sun_orientations)
        save_sun_orientations_to_file(sun_orientations, output_directory, sun_orientations_file_name)
        
    # Save the camera info to a text file in the specified directory
    save_camera_info_to_file(output_directory)

    # Save the scene ground truth data to a text file in the specified directory
    save_vals_to_file(output_directory, scene_gt_file_name, positions, rotations, object_id)

    # Save the initial inputs to a separate file
    save_rnd_gen_gt_info_to_file(output_directory, object_id, num_poses, sun_rnd_generated)
    
def generate_non_uniform_poses(num_frustums : int, num_points : int, max_dist : float, min_dist : float, 
                                fov : int, origin : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function generates non-uniform points with uneven density in a pyramid-shaped volume with
    specified parameters. The points are generated in a way that the density of points is higher
    closer to the origin of the frustum and lower further away from the origin.
    
    Returns a tuple containing two numpy arrays:
    `positions` and `rotations`.
    """
    
    # Generate points with uneven density
    num_points_per_frustum, remainder_points = divmod(num_points, num_frustums)
    positions = []
    rotations = []
    curr_dist = min_dist

    for i in range(num_frustums):
        curr_dist = min_dist + (i + 1) * (max_dist- min_dist) / num_frustums
        # Add an extra point to the first remainder_points frustums
        num_points = num_points_per_frustum + (i < remainder_points)
        section_points = generate_points_in_frustum(num_points, curr_dist, min_dist, fov, origin)
        positions.extend(section_points)
        rotations.extend([Quaternion.random().elements for _ in range(num_points)])

    positions = np.array(positions)
    rotations = np.array(rotations)
    
    return positions, rotations

def generate_points_in_frustum(num_points : int, max_dist : float, min_dist : float, fov : int, origin : np.ndarray) -> np.ndarray:
    half_base_angle = np.deg2rad(fov / 2)
    half_base_size = max_dist * np.tan(half_base_angle)

    inside_points = []
    num_generated_points = 0

    while len(inside_points) < num_points:
        x = np.random.uniform(-half_base_size, half_base_size)
        y = np.random.uniform(-half_base_size, half_base_size)
        z = np.random.uniform(min_dist, max_dist)
        point = np.array([x + origin[0], y + origin[1], z + origin[2]])
        
        # Check if the point is inside the frustum
        point_height = z
        frustum_height = max_dist
        frustum_ratio = point_height / frustum_height
        half_point_base_size = half_base_size * frustum_ratio

        if -half_point_base_size <= x <= half_point_base_size and -half_point_base_size <= y <= half_point_base_size:
            inside_points.append(point)
            num_generated_points += 1
            
    return np.array(inside_points)

def generate_random_sun_orientations(num_orientations: int):
    phi = 2 * np.pi * np.random.rand(num_orientations)  # azimuthal angle
    theta = np.arccos(2 * np.random.rand(num_orientations) - 1)  # polar angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])


def save_rnd_gen_gt_info_to_file(output_directory : str, group_id: str, num_poses : int,
                                 sun_rnd_generated : bool,
                                 min_distance : float = dc.min_distance, 
                                 max_distance : float = dc.max_distance, 
                                 num_frustums : int = dc.nb_layers,  
                                 sun_energy : float = dc.light_energy, 
                                 sun_dirs_file : str = dc.sun_orientations_file_name, 
                                 scene_info_file_name : str = dc.scene_info_file_name, 
                                 rnd_gen_mt_id : str = dc.random_poses_motion_id) -> None:
    
    with open(os.path.join(output_directory, scene_info_file_name), 'w') as f:
        f.write(f"----- Random poses info (ID : {rnd_gen_mt_id}) -----\n\n")
        
        f.write(f"Object Group ID: {group_id}\n")
        
        f.write(f"\n--- Pose generation info ---\n")
        f.write(f"Min distance (from camera): {min_distance} meters\n")
        f.write(f"Max distance (from camera): {max_distance} meters\n")
        f.write(f"Number of poses: {num_poses}\n")
        f.write(f"Number of generated layers: {num_frustums}\n")
        
        f.write(f"\n--- Lighting info ---\n")
        if sun_rnd_generated:
            f.write(f"Sun orientation : Random orientation for each pose ({sun_dirs_file})\n")
        else:
            f.write(f"Sun orientation [x, y, z] (unit vector): \n")
        f.write(f"Sun energy : {sun_energy} W/m^2\n")

def save_sun_orientations_to_file(sun_orientations : np.ndarray, output_directory : str, sun_orientations_file_name : str):
    with open(os.path.join(output_directory, sun_orientations_file_name), 'w') as f:
        for i, orientation in enumerate(sun_orientations):
            f.write(f"{i:05d},{orientation[0]:.6f},{orientation[1]:.6f},{orientation[2]:.6f}\n")

def save_vals_to_file(output_directory : str, output_file : str, positions : np.ndarray, orientations : np.ndarray, object_id : str):
    # Save the output array to a text file in the specified directory with the desired delimiter and format
    with open(os.path.join(output_directory, output_file), 'w') as f:
        for i, (orientation, position) in enumerate(zip(orientations, positions)):
            f.write(f"{i:05d},{object_id},{orientation[0]:.6f},{orientation[1]:.6f},{orientation[2]:.6f},{orientation[3]:.6f},{position[0]:.6f},{position[1]:.6f},{position[2]:.6f}\n")

if __name__ == "__main__":
    main()