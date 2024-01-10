import os
import numpy as np
import sys
import re
import math

import Utils.save_info_to_files_utils as save_utils
import Utils.dataset_constants as dc
import Utils.file_tools as file_tools





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
    print (f"maxc: {maxc}")
    print (f"minc: {minc}")

    # size of current object
    size = np.abs(np.array(maxc) - np.array(minc))
    effective_size = np.mean(size)
    print (f"effective_size: {effective_size}")
    # min object-camera distance
    min_distance = dc.calculate_distance_with_fov( 
        effective_size, dc.min_coverage_ratio
    )

    # max object-camera distance
    max_distance = dc.calculate_distance_with_fov(
        effective_size, dc.max_coverage_ratio
    )

    # place the corrners of the frustum
    apex_corners = dc.create_square_fov(min_distance)
    base_corners = dc.create_square_fov(max_distance)
    print (f"apex_corners: {apex_corners}")
    print (f"base_corners: {base_corners}")

    save_utils.html_motion_points(
        motion,
        np.vstack((apex_corners, base_corners)),
        object_name,
        os.path.join(parent_dir, f"{object_id}_{pose_ID}"),
        pose_ID,
    )
