"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Script to create a set of motions for multiple objects tumbling in space.
"""

import numpy as np
from pyquaternion import Quaternion
import os

from create_multi_obj_rnd_poses import save_vals_to_file, get_group_id, bbox_isolated, adjust_bounding_box
from tumble_motion_function import create_object_motion
import Utils.dataset_constants as dc
from Utils.save_info_to_files_utils import save_camera_info_to_file

#############################################
# Motion info

# Nb of motions to create
nb_motions: int = 15

# Starting motion id
start_tumble_id: str = "001"

# Minimum number of frames per motion
min_num_frames: int = 80

#############################################
# Object info

# Ids of objects for motion generation
object_ids: set[str] = {"02", "03", "04"}

#############################################
# Motion parameters

# Parameters for initial conditions [rad/s]
angular_velocity_mean : float = 0
angular_velocity_std : float = 0.70

# [m/s]
velocity_mean : float = 0
velocity_std : float = 3.25

#############################################
# Initial positions selection

xy_alpha : float = 1.75
xy_beta : float = 0.5

z_alpha : float = 0.75
z_beta : float = 4

#############################################
# Sun orientation

# Sun orientation (unit vector) [x, y, z]
sun_orientation : np.ndarray | None = None # None for random orientation

################################################
# Simulation properties

# Simulation duration [s]
sim_t = dc.default_sim_t

# Max distance from camera (in meters)
max_distance: int = 220
# Min distance from camera (in meters)
min_distance: int = 15

################################################
# Output properties

# Name of the output directory
proj_dir : str = "C:\\Users\\marin\\Documents\\BA5\\ProjB\\hubble"
input_dir : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")

################################################

def main():
    generate_motions(start_tumble_id, nb_motions, object_ids, 
                     angular_velocity_mean, angular_velocity_std, 
                     velocity_mean, velocity_std, sun_orientation, 
                     sim_t, max_distance, min_distance, output_directory)
    
def generate_motions(first_tumble_id: str, nb_motions : int, object_ids : set[str], 
                     angular_velocity_mean : float, angular_velocity_std : float, 
                     velocity_mean: float, velocity_std : float,
                     sun_orientation : np.ndarray | None, sim_t : float, 
                     max_distance : int, min_distance : int, out_directory : str) -> None:
    
    group_id = get_group_id(dc.group_ids, object_ids)
    if (group_id is None):
        raise ValueError("The group id could not be found")

    motions : list[str] = [f"{group_id}{int(first_tumble_id) + i:03d}" for i in range(nb_motions)]

    possible_start_positions = generate_non_uniform_positions(num_points=20*len(object_ids)*nb_motions, 
                                                              max_dist=max_distance, min_dist=min_distance, 
                                                              fov=dc.camera_fov, xy_alpha=xy_alpha, 
                                                              xy_beta=xy_beta, z_alpha=z_alpha, z_beta=z_beta)

    i : int = 0
    while i < nb_motions:
        motion : str = motions[i]
        print(f"Creating motion {motion}")
        # select random starting position
        p0 : dict[str, np.ndarray] = {}
        q0 : dict[str, Quaternion] = {}
        v0 : dict[str, np.ndarray] = {}
        w0 : dict[str, np.ndarray] = {}
        for object_id in object_ids:

            p0[object_id] = possible_start_positions[np.random.randint(0, len(possible_start_positions))]
            # select random starting velocity from 2 to 10 m/s
            v0[object_id] = np.random.normal(velocity_mean, velocity_std, 3)      
            #initial_velocity : np.ndarray = np.random.uniform(-max_tumble_velocity, max_tumble_velocity, 3)
            q0[object_id] = Quaternion.random()
            w0[object_id] = np.random.normal(angular_velocity_mean, angular_velocity_std, 3)
            #initial_angular_velocity : np.ndarray = np.random.uniform(-max_tumble_angle_velocity, max_tumble_angle_velocity, 3)

        if not quick_motion_check(p0, v0, min_num_frames//2, dc.frame_t, min_distance, max_distance, dc.camera_position, dc.camera_direction, dc.camera_fov):
            print("Motion is not valid")
            continue

        tumble_id : str = motion[-3:]
        print(f"Tumble id: {tumble_id}")

        positions : dict[str, np.ndarray] = {}
        orientations : dict[str, np.ndarray] = {} 

        for object_id in object_ids:
            positions[object_id], orientations[object_id] = create_object_motion(p0 = p0[object_id], v0 = v0[object_id], 
                                                        q0 = q0[object_id], w0 = w0[object_id], 
                                                        dt = dc.dt, sim_t = sim_t, frame_t = dc.frame_t, I = dc.object_inertia_matrices[object_id])

        nb_frames : dict[str, int] = get_nb_valid_frames(positions, orientations, dc.object_bounding_boxes, max_distance, min_distance, dc.camera_position, dc.camera_direction, dc.camera_fov)

        median_nb_frames : int = calculate_median_nb_frames(nb_frames)
        if (median_nb_frames < min_num_frames or 
            any(nb_frames[object_id] < min_num_frames // 2 for object_id in object_ids)):
            print("Motion is not valid")
            continue

        for object_id in object_ids:
            positions[object_id] = positions[object_id][:median_nb_frames]
            orientations[object_id] = orientations[object_id][:median_nb_frames]

        # Select random sun orientation in cartesian coordinates
        if sun_orientation is None:
            sun_orientation = generate_random_unit_vector()

        simulation_duration : float = median_nb_frames * dc.frame_t

        output_directory = os.path.join(out_directory, f"{group_id}{tumble_id}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        save_vals_to_file(output_directory, dc.scene_gt_file_name, positions, orientations)

        # Save scene info
        save_scene_info(output_directory, dc.scene_info_file_name, group_id, p0, q0, v0, w0, dc.dt, 
                        simulation_duration, dc.frame_t, median_nb_frames, tumble_id, sun_orientation, dc.light_energy)

        # Save the camera info
        save_camera_info_to_file(output_directory)

        i += 1
        
def generate_random_unit_vector() -> np.ndarray:
    phi = 2 * np.pi * np.random.random()  # azimuthal angle
    theta = np.arccos(2 * np.random.random() - 1)  # polar angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.array([x, y, z])

def calculate_median_nb_frames(nb_frames: dict[str, int]) -> int:
    sorted_frame_counts = sorted(nb_frames.values())
    half_index = len(sorted_frame_counts) // 2
    return (
        round(
            (
                sorted_frame_counts[half_index - 1]
                + sorted_frame_counts[half_index]
            )
            / 2
        )
        if len(sorted_frame_counts) % 2 == 0
        else sorted_frame_counts[half_index]
    )


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
    

def generate_non_uniform_positions(num_points : int, max_dist : float, min_dist : float, 
                                fov : int, xy_alpha : float = 2.5, xy_beta = 0.5, z_alpha : float = 0.8, z_beta : float = 10) -> np.ndarray:
    """
    This function generates non-uniform points with uneven density in a pyramid-shaped volume with
    specified parameters. The points are generated in the volume between `min_dist` and `max_dist`.
    The points are generated such that the density is higher closer to the camera in the z-axis and towards 
    the edges of the x and y axes. The points are generated in the camera's field of view.
    
    Returns a tuple containing two numpy arrays:
    `positions` and `rotations`.
    """
    
    # Generate points with uneven density
    half_base_angle = np.deg2rad(fov / 2)
    
    positions = np.zeros((num_points, 3))
    
    for i in range(num_points):
        z = np.random.beta(z_alpha, z_beta) * (max_dist - min_dist) + min_dist
        point_half_base_size = z * np.tan(half_base_angle)
        
        x, y = generate_non_uniform_points(1, xy_alpha, xy_beta, point_half_base_size, point_half_base_size)[0]
        positions[i] = np.array([x, y, z])
    
    return positions

def generate_non_uniform_points(nb_points : int, alpha : float = 2.5, beta : float = 0.5, max_x_val : float = 1.0, max_y_val : float = 1.0):
    '''
    Generates `nb_points` points on the square defined by [-`max_x_val`, `max_x_val`] x [-`max_y_val`, `max_y_val`].
    These points are generated non-uniformly, with a higher density to the edges of the square.
    '''
    n = max(nb_points, 10) # Make sure we can sample from both distributions
    
    # Generate samples
    x_samples1 = np.random.uniform(size=n)
    y_samples1 = np.random.beta(alpha, beta, size=n)

    x_samples2 = np.random.beta(alpha, beta, size=n)
    y_samples2 = np.random.uniform(size=n)

    # Concatenate, rescale, and shift samples
    x_samples = np.concatenate([x_samples1, x_samples2]) * 2 - 1
    y_samples = np.concatenate([y_samples1, y_samples2]) * 2 - 1

    x_samples = x_samples * np.choose(np.random.randint(2, size=n*2), [-1, 1]) * max_x_val
    y_samples = y_samples * np.choose(np.random.randint(2, size=n*2), [-1, 1]) * max_y_val

    # Stack x_samples and y_samples along a new second dimension
    points = np.column_stack((x_samples, y_samples))
    np.random.shuffle(points)

    return points[:nb_points]

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
        
        for object_id in sorted(object_ids):
            f.write(f"\n--- Object {object_id} info ---\n")
            f.write(f"\nInitial position [x, y, z] (m): "
                    f"{np.array2string(p0[object_id], separator=', ', precision=6)}\n")
            f.write(f"Initial attitude quaternion [qw, qx, qy, qz]: "
                    f"{np.array2string(np.array(q0[object_id].elements), separator=', ', precision=6)}\n")
            f.write(f"Initial velocity vector [x, y, z] (m/s): "
                    f"{np.array2string(v0[object_id], separator=', ', precision=6)}\n")
            f.write(f"Initial angular velocity vector [x, y, z] (rad/s): "
                    f"{np.array2string(w0[object_id], separator=', ', precision=6)}\n")

def quick_motion_check(positions : dict[str, np.ndarray], velocities : dict[str, np.ndarray], min_num_frames : int, 
                       frame_t : float, min_dist : float, max_dist : float, 
                       cam_pos : np.ndarray, cam_dir : np.ndarray, cam_fov : int) -> bool:
    ''' Checks if at least one object stays in the fov for the duration of the tumble. And if all objects are in the fov for at least half the tumble.'''
    
    one_object_stays_in_frustum = False
    for pos, velocity in zip(positions.values(), velocities.values()):
        new_1pos = pos + velocity * frame_t * min_num_frames/2
        new_2pos = pos + velocity * frame_t * min_num_frames
        if not is_in_frustum(new_1pos, max_dist, min_dist, cam_pos, cam_dir, cam_fov):
            return False
        if is_in_frustum(new_2pos, max_dist, min_dist, cam_pos, cam_dir, cam_fov):
            one_object_stays_in_frustum = True
    return one_object_stays_in_frustum
    
if __name__ == "__main__":
    main()