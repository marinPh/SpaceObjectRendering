import os
import numpy as np
import sys
import re
import math

import Utils.save_info_to_files_utils as save_utils
import Utils.dataset_constants as dc
import Utils.file_tools as file_tools


def create_square_fov(distance, fov):
    half_square = distance * math.sin(math.radians(dc.camera_fov / 2))
    square_corners = np.array(
        [
            [half_square, half_square, distance],
            [-half_square, half_square, distance],
            [-half_square, -half_square, distance],
            [half_square, -half_square, distance],
        ]
    )
    return square_corners


def calculate_distance_with_fov(
    effective_size, fov_degrees, image_dimension, coverage_ratio
):
    """
    Calculates the distance from the camera to the object based on the desired coverage ratio of the image,
    assuming a typical camera field of view.
    """
    fov_radians = math.radians(fov_degrees)
    covered_image_dimension = coverage_ratio * image_dimension
    distance = (effective_size / 2) / math.tan(fov_radians / 2)
    adjusted_distance = distance * (image_dimension / covered_image_dimension)
    return adjusted_distance


def show_motion(motion, object_name, parent_dir, pose_ID):
    print(f"parent_dir: {parent_dir}")
    object_id = object_name.split("_")[0]
    info_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "objects",
        "inertia",
        f"{object_name}_info.txt",
    )
   
    # convert the motion list to numpy array of float of shape (n,3)
    motion = np.array(motion, dtype=float)

    maxc, minc = file_tools.extract_corners(info_path)

    # size of current object
    size = np.abs(np.array(maxc) - np.array(minc))
    effective_size = np.mean(size)
    # min object-camera distance
    min_distance = calculate_distance_with_fov(
        effective_size, dc.camera_fov, 1024, 0.25**0.5
    )

    # max object-camera distance
    max_distance = calculate_distance_with_fov(
        effective_size, dc.camera_fov, 1024, 0.15**0.5
    )

    # place the corrners of the frustum
    apex_corners = create_square_fov(min_distance, dc.camera_fov)
    base_corners = create_square_fov(max_distance, dc.camera_fov)

    save_utils.html_motion_points(
        motion,
        np.vstack((apex_corners, base_corners)),
        object_name,
        os.path.join(parent_dir,"input", f"{object_id}_{pose_ID}"),
        pose_ID,
    )
