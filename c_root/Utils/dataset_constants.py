"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Constants for dataset generation.
"""

import numpy as np
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import file_tools


################################################
# Camera info

# Camera name
camera_name : str = "Camera"

# Camera position [x, y, z]
camera_position : np.ndarray = np.array([0, 0, 0])

# Camera orientation (quaternion) [w, x, y, z]
camera_direction : np.ndarray = np.array([0, 1, 0, 0])
# Parts of the program suppose that the camera is looking along the z-axis

# Camera field of view [degrees]
camera_fov : int = 40



################################################
# Lighting info

# Light name
light_name : str = "Light"

# Light energy [W/m^2]
light_energy  = 1000
possible_light_energies = light_energy*np.array([0.0100, 0.0117, 0.0137, 0.0161, 0.0189, 0.0221, 0.0259, 0.0304, 0.0356, 0.0418, 0.0489, 0.0574, 0.0672, 0.0788, 0.0924, 0.1083, 0.1269, 0.1487, 0.1743, 0.2043, 0.2395, 0.2807, 0.3290, 0.3857, 0.4520, 0.5298, 0.6210, 0.7279, 0.8532, 1.0000])



# Light position [x, y, z] only for visualization purposes (sun is infinitely far away)
light_position : np.ndarray = np.array([0, 0, 0])

# Light default orientation (unit vector) [x, y, z]
light_default_direction : np.ndarray = np.array([0, 0, 1])

################################################
# File names

# Scene ground truth file
scene_gt_file_name : str = "scene_gt.txt"

# Scene info file
scene_info_file_name : str = "scene_gt_info.txt"

# Sun orientations file
sun_orientations_file_name : str = "sun_gt.txt"

# Camera info file
camera_info_file_name : str = "scene_camera.txt"
# coverage for min distance
min_coverage_ratio = 0.25**0.5
# coverage for max distance
max_coverage_ratio = 0.05
# image dimension
image_dimension = 1024

################################################
# Output infos

# Constant number of columns per object in the motion file (8 for the current format) (objId(1), quaternion(4), pos(3))
NUM_COLS_PER_OBJECT : int = 8

# Name of progress log file
progress_log_file_name: str = "progress_log.txt"

# Name of the render folder
render_folder_name: str = "rgb"

# Name of segmentation folder
segmentation_folder_name: str = "seg"

# Name of the mask folder
mask_folder_name: str = "mask"

# Name of mask node in compositor
mask_node_name : str = "mask_output"

# Name of segmentation node in compositor
seg_node_name : str = "seg_output"

################################################
# Motion generation

# Simulation time step [s]
dt = 0.001
# Default (max) simulation duration [s]
default_sim_t = 40
# Time between frames [s]
frame_t = 0.1

#min wanted frames
min_frames = 100

################################################
# Random poses generation

# Number of poses for training set
train_num_poses : int = 3000

# Number of poses for validation set
val_num_poses : int = 750

# Number of layers of points to generate (less layers = more even distribution)
nb_layers : int = 300

# Random poses motion id
random_poses_motion_id : str = "000"

################################################
#distribution of the points in the frustum
def mean_and_covariance_matrix(object_name):
    min = calculate_distance_with_fov(object_size(object_name), min_coverage_ratio)
    max = calculate_distance_with_fov(object_size(object_name), max_coverage_ratio)
    mean = np.array([0,0,(max-min)/2 + min])
    #get distance of origin from fov limit
    side_limit = np.tan(np.deg2rad(camera_fov/2))*mean[2]
    good_var = side_limit/10
    covariance_matrix = np.array([[good_var,0,0],[0,good_var,0],[0,0,(max-min)/16]])
    return mean, covariance_matrix/5
    


################################################

# Objects in project

# Object names and ids
object_names : dict[str, str] = {
    "01": "01_James_Webb_Space_Telescope",
    "02": "02_Hubble_Space_Telescope",
    "03": "03_CosmosLink",
    "04": "04_Rocket_Body",
    "5": "5_MTM"
}

# Object bounding boxes
object_bounding_boxes: dict[str, np.ndarray] = {
    "01": np.array([[-6.4353, -10.0689, -3.4674], [6.4153, 10.9273, 7.7901]]),
    "02": np.array([[-12.2696, -6.5803, -6.6335], [5.7087, 6.6909, 6.5916]]),
    "03": np.array([[-7.6050, -3.0449, -1.4568], [7.6168, 3.7835, 1.5087]]),
    "04": np.array([[-5.7771, -2.0733, -2.1503], [5.4408, 2.0764, 2.1112]])
}
# Object inertia matrices (inertia tensor)
object_inertia_matrices: dict[str, np.ndarray] = {
    "01": np.array([[119734.71,  0.00,   0.00], [  0.00, 44900.79,   0.00], [  0.00,   0.00, 133802.82]]),
    "02": np.array([[27453.70,   0.00,   0.00], [  0.00, 195444.31,   0.00], [  0.00,   0.00, 196930.26]]),
    "03": np.array([[11850.91,   0.00,   0.00], [  0.00, 4407.01,   0.00], [  0.00,   0.00, 13850.67]]),
    "04": np.array([[11755.42,   0.00,   0.00], [  0.00, 29069.82,   0.00], [  0.00,   0.00, 28807.09]])
}

# Object groups
group_ids : dict[str, set[str]] = {
    "01": {"01"}, # --
    "02": {"02"}, # --
    "03": {"03"}, # --
    "04": {"04"}, # --
    "21": {"01", "02"}, # --
    "22": {"01", "03"}, # --
    "23": {"01", "04"},
    "24": {"02", "03"},
    "25": {"02", "04"}, # --
    "26": {"03", "04"},
    "31": {"01", "02", "03"},
    "32": {"01", "02", "04"},
    "33": {"01", "03", "04"}, # --
    "34": {"02", "03", "04"}, # --
    "41": {"01", "02", "03", "04"} # --
}

def calculate_distance_with_fov(
    effective_size, coverage_ratio
):
    """
    Calculates the distance from the camera to the object based on the desired coverage ratio of the image,
    assuming a typical camera field of view.
    """
    fov_radians = math.radians(camera_fov)
    covered_image_dimension = coverage_ratio * image_dimension
    distance = (effective_size / 2) / math.tan(fov_radians / 2)
    adjusted_distance = distance * (image_dimension / covered_image_dimension)
    return adjusted_distance

def create_square_fov(distance):
    half_square = distance * math.sin(math.radians(camera_fov / 2))
    square_corners = np.array(
        [
            [half_square, half_square, distance],
            [-half_square, half_square, distance],
            [-half_square, -half_square, distance],
            [half_square, -half_square, distance],
        ]
    )
    return square_corners

def object_size(object_name):
    # go look for in the file objects/inertia/{object_name}_info.txt
    # return the size of the object
    # if the file doesn't exist, return 0
    info_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "objects",
        "inertia",
        f"{object_name}_info.txt",
    )
    if os.path.exists(info_path):
        maxc, minc = file_tools.extract_corners(info_path)
        size = np.abs(np.array(maxc) - np.array(minc))
        effective_size = np.mean(size)
        return effective_size
    else:
        return 0




