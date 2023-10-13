"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Generates random poses for multiple objects.
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from create_random_poses import generate_non_uniform_poses, save_rnd_gen_gt_info_to_file, save_sun_orientations_to_file, generate_random_sun_orientations
from Utils.save_info_to_files_utils import save_camera_info_to_file
import Utils.dataset_constants as dc

################################################
# User-defined inputs

# Output directory
proj_dir : str = "C:\\Users\\marin\\Documents\\BA5\\ProjB\\hubble"
input_dir : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")

# Number of poses to generate
num_poses = dc.val_num_poses

# Ids of objects for pose generation
object_ids: set[str] = {"01", "02"}
'''
# Object names and their corresponding object ID
"01": "James Webb Space Telescope",
"02": "Hubble Space Telescope",
"03": "CosmosLink",
"04": "Rocket Body"
'''

# Whether the sun orientation is randomly generated or not
sun_rnd_generated : bool = True

################################################

def main() -> None:
    group_id = get_group_id(dc.group_ids, object_ids)
    if group_id is None:
        raise ValueError("Group id corresponding to select object_ids did not find")
    output_directory : str = os.path.join(out_directory, f"{group_id}{dc.random_poses_motion_id}")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    rnd_positions: dict[str, np.ndarray] = {}
    rnd_orientations: dict[str, np.ndarray] = {}
    for obj_id in object_ids:
        rnd_positions[obj_id], rnd_orientations[obj_id] = generate_non_uniform_poses(
            num_frustums=dc.nb_layers, num_points=num_poses*2,
            max_dist=dc.max_distance, min_dist=dc.min_distance,
            fov=dc.camera_fov, origin=dc.camera_position
        )
        idx = np.random.permutation(np.arange(len(rnd_positions[obj_id])))
        rnd_positions[obj_id] = rnd_positions[obj_id][idx]
        rnd_orientations[obj_id] = rnd_orientations[obj_id][idx] # Shuffle the poses

    positions: dict[str, np.ndarray] = {}
    orientations: dict[str, np.ndarray] = {}
    for obj_id in object_ids:
        positions[obj_id] = np.zeros((num_poses, 3))
        orientations[obj_id] = np.zeros((num_poses, 4))

    for i in tqdm(range(num_poses)):
        b_boxes: list[np.ndarray] = []
        for obj_id in np.random.permutation(list(object_ids)): # Shuffle the object ids to ensure that the poses are evenly distributed
            j : int = 0
            while True:  # Loop until a valid pose is found
                pos, ori = rnd_positions[obj_id][j], rnd_orientations[obj_id][j]
                adjusted_bbox : np.ndarray = adjust_bounding_box(dc.object_bounding_boxes[obj_id], pos, ori)
                if bbox_isolated(b_boxes, adjusted_bbox):  # If there's no overlap, it's a valid pose
                    b_boxes.append(adjusted_bbox)
                    positions[obj_id][i] = pos
                    orientations[obj_id][i] = ori
                    rnd_positions[obj_id] = np.delete(rnd_positions[obj_id], j, axis=0) # Remove the pose from the list of poses
                    rnd_orientations[obj_id] = np.delete(rnd_orientations[obj_id], j, axis=0)
                    break  # Break the loop once a valid pose is found
                j += 1
                if j >= len(rnd_positions[obj_id]):  # If we've exhausted all poses
                    print(f"Could not find a non-overlapping bounding box for object {obj_id}")
                    positions[obj_id][i] = None
                    orientations[obj_id][i] = None
                    break

    if sun_rnd_generated:
        sun_orientations = generate_random_sun_orientations(num_poses)
        sun_orientations = np.array(sun_orientations)
        save_sun_orientations_to_file(sun_orientations, output_directory, dc.sun_orientations_file_name)

    save_vals_to_file(output_directory, dc.scene_gt_file_name, positions, orientations)

    save_camera_info_to_file(output_directory)

    save_rnd_gen_gt_info_to_file(output_directory, group_id, num_poses, sun_rnd_generated)
    

def get_group_id(group_ids: dict[str, set[str]], object_ids: set[str]) -> str | None:
    return next(
        (group_id for group_id, ids in group_ids.items() if ids == object_ids),
        None,
    )

def adjust_bounding_box(bbox_corners: np.ndarray, 
                        position: np.ndarray, 
                        orientation: np.ndarray) -> np.ndarray:
    '''Adjust the bounding box to the object's position and orientation (quaternion) [qw, qx, qy, qz], transform the bounding box format from corners to center and axes'''

    # Convert corners to center and axes
    center_at_origin = np.mean(bbox_corners, axis=0)
    axes_lengths = (bbox_corners[1] - bbox_corners[0]) / 2

    # Define unit vectors in x, y, z directions
    unit_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Scale unit vectors to the size of each axis
    axes_at_origin = unit_vectors * axes_lengths


    # Convert quaternion from scalar first to scalar last format
    orientation = np.roll(orientation, -1)
    # Convert the quaternion to a rotation matrix
    r : R = R.from_quat(orientation) # !!! Takes scalar last format for quaternions [x, y, z, w]
    rotation_matrix : np.ndarray = r.as_matrix()

    # Rotate the axes
    rotated_axes = np.dot(rotation_matrix, axes_at_origin.T).T

    # Translate the center
    new_center = center_at_origin + position

    return np.vstack((new_center, rotated_axes))


def separating_axis_check(l: float, ra: float, rb: float) -> bool:
    """
    Check if the absolute difference between the lengths of two projections is 
    larger than the sum of the projections.
    """
    return abs(l) > ra + rb

def bbox_isolated(b_boxes: list[np.ndarray], 
                  new_bbox: np.ndarray) -> bool:
    """
    Check if a bounding box is isolated from a list of bounding boxes.
    """
    return not any(bbox_overlaps(bbox, new_bbox) for bbox in b_boxes)

def bbox_overlaps(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Check if two bounding boxes overlap using the Separating Axis Theorem.
    """
    axes = np.zeros((15, 3))  # Define a matrix to hold the 15 candidate separating axes
    axes[0:3, :] = box1[1:4, :]  # Axes of box1
    axes[3:6, :] = box2[1:4, :]  # Axes of box2
    axes[6:9, :] = np.cross(box1[1:4, :], box2[1, :])  # Cross product of axes from box1 and box2
    axes[9:12, :] = np.cross(box1[1:4, :], box2[2, :])
    axes[12:15, :] = np.cross(box1[1:4, :], box2[3, :])

    for i in range(axes.shape[0]):
        l = np.dot(box2[0, :] - box1[0, :], axes[i, :])  # Projection of the vector between box centers onto the candidate axis
        ra = sum(abs(np.dot(box1[j, :], axes[i, :])) for j in range(1, 4))  # Sum of the projections of box1's axes onto the candidate axis
        rb = sum(abs(np.dot(box2[j, :], axes[i, :])) for j in range(1, 4))  # Sum of the projections of box2's axes onto the candidate axis
        
        if separating_axis_check(l, ra, rb):  # If the projections do not overlap, there is a separating axis and the boxes do not intersect
            return False
    
    return True  # If no separating axis was found, the boxes intersect

def save_vals_to_file(output_directory : str, output_file : str, 
                      positions : dict[str, np.ndarray], 
                      orientations : dict[str, np.ndarray]) -> None:
    # Save the output array to a text file in the specified directory with the desired delimiter and format
    with open(os.path.join(output_directory, output_file), 'w') as f:
        # Get all object IDs
        object_ids : list[str] = list(positions.keys())
        if object_ids != list(orientations.keys()):
            raise ValueError("The object IDs in the positions and orientations dictionaries do not match")
        
        object_ids.sort()  # To ensure consistent order

        # Get the number of frames from the first object ID
        num_frames : int = len(positions[object_ids[0]])
        if any(len(positions[object_id]) != num_frames 
               and len(orientations[object_id] != num_frames) for object_id in object_ids):
            raise ValueError("The number of frames in the positions and orientations dictionaries do not match")

        for frame_id in range(num_frames):
            f.write(f"{frame_id:05d}")  # Write the frame ID, padded with zeros on the left

            for object_id in object_ids:
                # Retrieve the orientation and position of the object for this frame, or None if they do not exist
                orientation : np.ndarray = orientations[object_id][frame_id]
                position : np.ndarray = positions[object_id][frame_id]

                # If the orientation or position are None, replace them with 'NaN'
                orientation_str : list[str] = [f'{x:.6f}' if x is not None else 'NaN' for x in orientation]
                position_str : list[str] = [f'{x:.6f}' if x is not None else 'NaN' for x in position]

                # Write the object ID, orientation, and position to the file
                f.write(f",{object_id},{','.join(orientation_str)},{','.join(position_str)}")

            # Add a newline at the end of each frame
            f.write("\n")
    
if __name__ == '__main__':
    main()