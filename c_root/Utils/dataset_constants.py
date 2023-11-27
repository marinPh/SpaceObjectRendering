"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Constants for dataset generation.
"""

import numpy as np

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

# Minimum distance from camera to an object [m]
min_distance : float = 20

# Maximum distance from camera to an object [m]
max_distance : float = 300

################################################
# Lighting info

# Light name
light_name : str = "Light"

# Light energy [W/m^2]
light_energy : float = 1360

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
################################################