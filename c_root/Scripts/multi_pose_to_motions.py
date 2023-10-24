"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Generates a motion for multiple objects tumbling in space.
"""
from __future__ import annotations

import argparse

import numpy as np
from pyquaternion import Quaternion
import math
import os

from pose_to_motion import save_vals_to_file, get_group_id, bbox_isolated, adjust_bounding_box
from tumble_motion_function import create_object_motion
import Utils.dataset_constants as dc
from Utils.save_info_to_files_utils import save_camera_info_to_file

#############################################
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py arg1 arg2")
else:
    arg1= sys.argv[1]
    arg2 = sys.argv[2]
    print(f"Argument 1: {arg1}")
    print(f"Argument 2: {arg2}")
# Motion info
main_obj_name = arg1
pose_id = arg2

# Tumble id
tumble_id: str =pose_id
#############################################
# Object info

# Ids of objects for motion generation
object_ids: set[str] = {str(main_obj_name).partition('_')[0]}


#############################################
# Initial conditions

# Initial position [m] [x,y,z]
p0 : dict[str, np.ndarray] = {
    "01" : np.array([15, 0, 70]),
    "02" : np.array([15, 0, 40])
}
# Initial attitude quaternion [q0,qx,qy,qz]
q0 : dict[str, Quaternion] = {
    "01" : Quaternion(1, 0, 0, 0),
    "02" : Quaternion(1, 0, 0, 0)
}
# Initial velocity vector [m/s] [x,y,z]
v0 : dict[str, np.ndarray] = {
    "01" : np.array([-3.5, 0, -5]),
    "02" : np.array([-3.5, 0, 5])
}
# Initial rotation vector [rad/s] [x,y,z]
w0 : dict[str, np.ndarray] = {
    "01" : np.array([math.pi/8, 0, 0]),
    "02" : np.array([math.pi/8, 0, 0])
}

#############################################
# Sun orientation

# Select random sun orientation in cartesian coordinates
sun_orientation : np.ndarray | None = None # None for random orientation

################################################
# Simulation properties

# Simulation duration [s]
sim_t = dc.default_sim_t

# Max distance from camera (in meters)
max_distance: int = 175 # dc.max_distance
# Min distance from camera (in meters)
min_distance: int = 5 # dc.min_distance

################################################
# Output properties

# Name of the output directory
proj_dir : str = os.path.dirname(os.path.dirname(__file__))
input_directory : str = os.path.join(proj_dir,"input")
output_directory : str = os.path.join(proj_dir,"output")

################################################

def main():
    create_motions(tumble_id, object_ids, p0, q0, v0, w0, sun_orientation, sim_t, max_distance, min_distance, output_directory)
    
def create_motions(tumble_id: str, object_ids : set[str], p0 : dict[str, np.ndarray], q0 : dict[str, Quaternion], v0 : dict[str, np.ndarray], w0 : dict[str, np.ndarray], sun_orientation : np.ndarray | None, sim_t : float, max_distance : int, min_distance : int, out_directory : str) -> None:
    
    positions : dict[str, np.ndarray] = {}
    orientations : dict[str, np.ndarray] = {} 

    for object_id in np.random.permutation(list(object_ids)): 
    # Randomize the order of the objects so that the first object is not always the first in the list
        positions[object_id], orientations[object_id] = create_object_motion(p0 = p0[object_id], v0 = v0[object_id], 
                                                    q0 = q0[object_id], w0 = w0[object_id], 
                                                    dt = dc.dt, sim_t = sim_t, frame_t = dc.frame_t, I = dc.object_inertia_matrices[object_id])
    
    nb_frames : dict[str, int] = get_nb_valid_frames(positions, orientations, dc.object_bounding_boxes, max_distance, min_distance, dc.camera_position, dc.camera_direction, dc.camera_fov)
    
    # Select random sun orientation in cartesian coordinates
    if sun_orientation is None:
        sun_orientation = generate_random_unit_vector()
    
    max_nb_frames : int = max(nb_frames.values())
    for object_id in object_ids:
        range_size = max_nb_frames - nb_frames[object_id]
        positions[object_id][nb_frames[object_id]:max_nb_frames] = np.full((range_size, 3), np.nan)
        orientations[object_id][nb_frames[object_id]:max_nb_frames] = np.full((range_size, 4), np.nan)
        positions[object_id] = positions[object_id][:max_nb_frames]
        orientations[object_id] = orientations[object_id][:max_nb_frames]
    
    simulation_duration : float = max_nb_frames * dc.frame_t
    
    group_id = get_group_id(dc.group_ids, object_ids)
    if (group_id is None):
        raise ValueError("The group id could not be found")
    
    output_directory = os.path.join(out_directory, f"{group_id}{tumble_id}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    save_vals_to_file(output_directory, dc.scene_gt_file_name, positions, orientations)
    
    # Save scene info
    save_scene_info(output_directory, dc.scene_info_file_name, group_id, p0, q0, v0, w0, dc.dt, 
                    simulation_duration, dc.frame_t, max_nb_frames, tumble_id, sun_orientation, dc.light_energy)

    # Save the camera info
    save_camera_info_to_file(output_directory)
        
def generate_random_unit_vector() -> np.ndarray:
    phi = 2 * np.pi * np.random.random()  # azimuthal angle
    theta = np.arccos(2 * np.random.random() - 1)  # polar angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.array([x, y, z])

def get_nb_valid_frames(positions: dict[str, np.ndarray], 
                  orientations: dict[str, np.ndarray], 
                  bounding_boxes: dict[str, np.ndarray], 
                  max_distance: int, 
                  min_distance: int, 
                  cam_p: np.ndarray, 
                  cam_o: np.ndarray, 
                  cam_fov: int) -> dict[str, int]:
    '''Get the number of frames in the motion for each object'''
    
    nb_frames : dict[str, int] = {}

    num_values = len(next(iter(positions.values()))) # Number of frames in the motion

    b_boxes = np.empty((num_values,), dtype=object) # List of bounding boxes for each frame
    b_boxes[:] = [[] for _ in range(num_values)]
    
    frames_before_collision = num_values

    for object_id, obj_positions, obj_orientations in zip(positions.keys(), positions.values(), orientations.values()):
        nb_frames[object_id] = 0
        for i in range(len(obj_positions)):
            # Get the adjusted bounding box
            pos = obj_positions[i]
            ori = obj_orientations[i] # Convert to numpy array from quaternion
            adjusted_bbox = adjust_bounding_box(bounding_boxes[object_id], 
                                                pos, ori)
            # Check if the object is in the camera's field of view
            if not is_in_fov(adjusted_bbox, max_distance, min_distance, cam_p, cam_o, cam_fov):
                break
            if not bbox_isolated(b_boxes[i], adjusted_bbox):
                frames_before_collision = min(frames_before_collision, i)
                break
            b_boxes[i].append(adjusted_bbox)
            nb_frames[object_id] += 1
    nb_frames = {k: min(v, frames_before_collision) for k, v in nb_frames.items()}
    print("Nb of frames :", nb_frames)
    return nb_frames

def is_in_fov(adjusted_bbox: np.ndarray, 
              max_distance: int, 
              min_distance: int, 
              cam_p: np.ndarray, 
              cam_o: np.ndarray, 
              cam_fov: int) -> bool:
    '''Check if any part of the object is in the camera's field of view'''

    # Get all corners of the bounding box
    corners = get_corners(adjusted_bbox)

    return any(
        is_in_frustum(
            corner, max_distance, min_distance, cam_p, cam_o, cam_fov
        )
        for corner in corners
    )
    
    
def get_corners(bbox : np.ndarray) -> np.ndarray:
    '''Get all corners of the oriented bounding box'''

    # Get the center and half-lengths along each principal axis of the bounding box
    center : np.ndarray = bbox[0, :]
    half_length1 : np.ndarray = bbox[1, :]
    half_length2 : np.ndarray = bbox[2, :]
    half_length3 : np.ndarray = bbox[3, :]

    # Create the corners
    corners : np.ndarray = np.array([center - half_length1 - half_length2 - half_length3,
                                     center - half_length1 - half_length2 + half_length3,
                                     center - half_length1 + half_length2 - half_length3,
                                     center - half_length1 + half_length2 + half_length3,
                                     center + half_length1 - half_length2 - half_length3, 
                                     center + half_length1 - half_length2 + half_length3, 
                                     center + half_length1 + half_length2 - half_length3, 
                                     center + half_length1 + half_length2 + half_length3])

    return corners

def is_in_frustum(point: np.ndarray, 
                  max_distance: float, 
                  min_distance: float, 
                  cam_p: np.ndarray, 
                  cam_o: np.ndarray, 
                  cam_fov: int) -> bool:
    '''Check if a point is in the camera's field of view'''
    
    if not np.array_equal(cam_o, [0, 1, 0, 0]):
        raise NotImplementedError("Only the default camera orientation Quaternion(0, 1, 0, 0) is supported")
    
    x = point[0] - cam_p[0]
    y = point[1] - cam_p[1]
    z = point[2] - cam_p[2]

    half_base_angle = np.deg2rad(cam_fov / 2)
    half_base_size = max_distance * np.tan(half_base_angle)
    
    # Check if the point is inside the frustum
    frustum_height = max_distance - min_distance
    frustum_ratio = z / frustum_height
    half_point_base_size = half_base_size * frustum_ratio

    return  (-half_point_base_size <= x <= half_point_base_size 
             and -half_point_base_size <= y <= half_point_base_size
             and min_distance <= z <= max_distance)

def save_scene_info(output_directory : str, file_name : str, group_id : str,
                    p0 : dict[str, np.ndarray], q0 : dict[str, Quaternion], 
                    v0 : dict[str, np.ndarray], w0 : dict[str, np.ndarray],
                    dt : float, sim_t : float, frame_t : float, 
                    num_frames : int, tumble_id : str,
                    sun_rot : np.ndarray, sun_energy : float) -> None:
    
    object_ids = list(p0.keys())
    if p0.keys() != q0.keys() or p0.keys() != v0.keys() or p0.keys() != w0.keys():
        raise ValueError("Keys of initial parameters do not match!!")
    
    with open(os.path.join(output_directory, file_name), 'w') as f:
        f.write(f"----- Tumble {tumble_id} info -----\n\n")
        
        f.write(f"Object Group ID: {group_id}\n")
        
        f.write(f"\n--- Simulation info ---\n")
        f.write(f"Simulation time step: {dt} s\n")
        f.write(f"Simulation duration: {sim_t} s\n")
        f.write(f"Time between frames: {frame_t} s\n")
        f.write(f"Number of frames: {num_frames}\n")
        
        f.write(f"\n--- Lighting info ---\n")
        f.write(f"Sun orientation [x, y, z] (unit vector): {np.array2string(sun_rot, separator=', ', precision=6)}\n")
        f.write(f"Sun energy: {sun_energy} W/m^2\n")
        
        for object_id in object_ids:
            f.write(f"\n--- Object {object_id} info ---\n")
            f.write(f"\nInitial position [x, y, z] (m): "
                    f"{np.array2string(p0[object_id], separator=', ', precision=6)}\n")
            f.write(f"Initial attitude quaternion [qw, qx, qy, qz]: "
                    f"{np.array2string(np.array(q0[object_id].elements), separator=', ', precision=6)}\n")
            f.write(f"Initial velocity vector [x, y, z] (m/s): "
                    f"{np.array2string(v0[object_id], separator=', ', precision=6)}\n")
            f.write(f"Initial angular velocity vector [x, y, z] (rad/s): "
                    f"{np.array2string(w0[object_id], separator=', ', precision=6)}\n")

def quick_motion_check(pos : np.ndarray, velocity : np.ndarray, min_num_frames : int, 
                       frame_t : float, min_dist : float, max_dist : float, 
                       cam_pos : np.ndarray, cam_dir : np.ndarray, cam_fov : int) -> bool:

    pos = pos + velocity * frame_t * min_num_frames

    return is_in_frustum(pos, max_dist, min_dist, cam_pos, cam_dir, cam_fov)
    
if __name__ == "__main__":
    main()