"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Applies the motion from the dataset to the objects in the scene.
"""
from __future__ import annotations

import argparse

import bpy
import numpy as np
from mathutils import Vector
import os
import re
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Utils import dataset_constants as dc






################################################

# Path for the file that contains the position data
# File should be formatted as follows:
# frame_nb, object_id1, q10, q1x, q1y, q1z, pos1x, pos1y, pos1z, object_id2, q20, q2x, q2y, q2z, pos2x, pos2y, pos2z, ...
proj_dir: str = os.path.dirname(os.path.dirname(__file__))
# Output directory
input_directory: str = os.path.abspath("input")
output_directory: str = os.path.abspath("output")


# Motion ID
print(f"---> {sys.argv}")
sys_argv = sys.argv
if len(sys_argv) < 3:
    print("Usage: python script.py arg1 arg2")
else:
    main_obj_name = sys_argv[-2]
    pose_id = sys_argv[-1]
# Motion info



blend_file_path = os.path.join(proj_dir, "objects", "blend", f"{main_obj_name}.blend")

# TODO check if this does the right motion ID
motion_id = main_obj_name.split('_')[0] + pose_id

# Path for the file that contains the position and orientation data
motions_path = os.path.join(input_directory, motion_id, dc.scene_gt_file_name)

# Path for the file that contains the sun direction data
sun_str = os.path.join(input_directory, motion_id, "sun_gt.txt")
sun_path = os.path.abspath(sun_str)

# Path for the file that contains info about the scene (camera, light, etc.)
info_path = os.path.join(input_directory, motion_id, dc.scene_info_file_name)

# Light direction (if sun_path is None)
light_dir = dc.light_default_direction  # (1, 0, 0)

# Number of images to render (set to None for rendering all images from motion) Otherwise, renders the first nb_im images
nb_im = None


################################################


def apply_blender_animation(objects_dict: dict[str, str],
                            motions_path: str,
                            sun_path: str | None,
                            info_path: str,
                            num_cols_per_object: int,
                            lightsource_name: str,
                            camera_name: str,
                            cam_pos: Vector,
                            cam_rot: np.ndarray,
                            light_pos: Vector,
                            light_rot: np.ndarray,
                            light_energy: float,
                            nb_im: int | None) -> int:
    # Import motion and sun data 
    motion_quat, trans_vec, objects_ids = motion_and_translation_import(motions_path, list(objects_dict.keys()),
                                                                        num_cols_per_object)

    sun_rot = None
    if sun_path is not None:
        sun_rot = sun_vectors_to_quaternions(sun_direction_import(sun_path))
        print("Sun rotation shape:", sun_rot.shape)
    elif info_path is not None:
        possible_light_rot = np.array(scene_info_import(info_path))
        # Check if light rotation is not empty
        if possible_light_rot is not None and len(possible_light_rot) > 0:
            light_rot = possible_light_rot

    # Initialize objects
    init_objects(objects_dict, objects_ids)
    init_scene(camera_name, lightsource_name, cam_pos, cam_rot, light_pos, light_rot, light_energy)

    create_animation(objects_dict, objects_ids, motion_quat, trans_vec, lightsource_name, sun_rot, nb_im=nb_im)

    return len(motion_quat[objects_ids[0]])


def motion_and_translation_import(mpath: str, objects_ids: list[str], NUM_COLS_PER_OBJECT: int = 8) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    # import quaternion data from .txt file
    with open(mpath) as f:
        first_line = f.readline().replace("\n", "").replace(" ", "")
        print("\nFirst line from motion path :", first_line.split(","))
        num_cols = len(first_line.split(","))

    motion_quat = {}
    motion_trans = {}
    objects_in_file_ids: list[str] = []
    frames = np.loadtxt(mpath, usecols=0, delimiter=",")
    if not np.all(np.diff(frames) == 1) or frames[0] != 0:
        raise ValueError("Frames are not consecutive or do not start at 0")

    for i in range(1, num_cols, NUM_COLS_PER_OBJECT):
        """
        if first_line.split(",")[i] not in objects_ids:
            fl = first_line.split(",")[i]
            raise ValueError(f"Object ID {fl, objects_ids} not recognized")"""

        motion_quat[first_line.split(",")[i]] = np.loadtxt(mpath, usecols=range(i + 1, i + 5), delimiter=",")
        motion_trans[first_line.split(",")[i]] = np.loadtxt(mpath, usecols=range(i + 5, i + 8), delimiter=",")
        objects_in_file_ids.append(first_line.split(",")[i])

    return motion_quat, motion_trans, objects_in_file_ids


def sun_direction_import(sunpath: str) -> np.ndarray:
    if os.path.exists(sunpath):
        return np.loadtxt(sunpath, usecols=range(1, 4), delimiter=",")
    else:
        raise FileNotFoundError(f"The file {sunpath} does not exist.")


def scene_info_import(info_path: str) -> tuple[float, float, float]:
    # Initialize sun_orientation list
    sun_orientation: tuple[float, float, float] | None = None
    with open(info_path, "r") as file:
        # Iterate through each line in the file
        for line in file:
            if "Sun orientation" in line:
                if not (
                        matches := re.findall(
                            r'[-+]?\d+(?:\.\d+)?', line  # Regex to extract sun orientation values
                        )
                ):
                    raise ValueError(
                        f"Could not parse sun orientation from line: {line}"
                    )
                # Extracted values as floats
                sun_orientation = tuple(float(match) for match in matches)
                break
    if sun_orientation is None:
        raise ValueError("No sun orientation found in file")
    print("Sun orientation:", sun_orientation)
    return sun_orientation


# Transform sun direction vectors to Quaternions
def sun_vectors_to_quaternions(sun_vectors: np.ndarray) -> np.ndarray:
    quaternions = np.zeros((sun_vectors.shape[0], 4))
    for i, sun_vector in enumerate(sun_vectors):
        # Normalize the sun vector
        norm = np.linalg.norm(sun_vector)
        if norm == 0:
            quaternions[i] = np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion for zero vector
        else:
            sun_vector = sun_vector / norm  # Normalize the vector

            # Forward direction in the target orientation
            forward = np.array([0, 0, -1])

            # Compute the cosine of the angle (dot product with forward)
            cos_theta = np.dot(sun_vector, forward)

            # Compute the rotation axis
            axis = np.cross(forward, sun_vector)
            axis_norm = np.linalg.norm(axis)
            if axis_norm != 0:
                axis /= axis_norm

            theta = np.arccos(cos_theta)
            quaternion = np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * axis])
            quaternions[i] = quaternion
    return quaternions


def apply_sun_quaternion(sun_obj, sun_quaternion, frame_num):
    if sun_quaternion is not None:
        sun_obj.rotation_mode = 'QUATERNION'
        sun_obj.rotation_quaternion = sun_quaternion[frame_num]
        sun_obj.keyframe_insert(data_path="rotation_quaternion", index=-1)


def create_animation(objects_dict: dict[str, str], object_ids: list[str], motion_quat: dict, trans_vec: dict,
                     lightsource_name: str = "Lightsource", sun_dir: np.ndarray | None = None,
                     nb_im: int | None = None) -> None:
    # generating motion and rendering

    # Either render all images in motion or only the first nb_im images
    length = motion_quat[object_ids[0]].shape[0] if nb_im is None else nb_im  # Number of images to render
    print(f'Rendering {length} images')

    # 
    for frame_num in range(length):
        # for every frame
        bpy.context.scene.frame_set(frame_num)

        # insert sun direction
        apply_sun_quaternion(bpy.data.objects[lightsource_name], sun_dir, frame_num)

        insert_object_keyframes(objects_dict, main_obj_name.split("_")[0], motion_quat, trans_vec, frame_num)

        frame_num += 1


def insert_object_keyframes(objects_dict: dict[str, str], object_id: str, motion_quat: dict, trans_vec: dict,
                            frame_nb: int) -> None:
    obj = bpy.data.objects[main_obj_name]
    # change model position
    obj.location = trans_vec[object_id][frame_nb, :]
    # insert changes in position into keyframe
    obj.keyframe_insert(data_path="location", index=-1)
    # change model orientation
    obj.rotation_quaternion = motion_quat[object_id][frame_nb, :]
    # insert changes in orientation into keyframe
    obj.keyframe_insert(data_path="rotation_quaternion", index=-1)


def init_objects(objects_dict: dict[str, str], object_ids: list[str]):
    # Clear animation data and constraints for all objects
    for obj in bpy.data.objects:
        obj.animation_data_clear()
        obj.constraints.clear()

    id = main_obj_name.split("_")[0]
    print(f"Initializing object {id}")
    print(f"keys: {objects_dict.keys()}")
    print(f"Object name: {main_obj_name}")
    target = bpy.data.objects[main_obj_name]
    target.rotation_mode = "QUATERNION"
    target.hide_render = id not in object_ids  # Set final render
    target.location = (np.nan, np.nan, np.nan)  # Set location to NaN
    target.rotation_quaternion = (np.nan, np.nan, np.nan, np.nan)  # Set rotation to NaN


def init_scene(camera_name: str, light_name: str, cam_pos: Vector,
               cam_rot: np.ndarray, light_pos: Vector,
               light_rot: np.ndarray, light_energy: float):
    # Set camera position and rotation
    init_camera(camera_name, cam_pos, cam_rot)
    init_sun(light_name, light_pos, light_rot, light_energy)

#TODO recheck the inputs of camera FOV FOCAL LENGTH and SENSOR WIDTH
def init_camera(camera_name: str, cam_pos: Vector, cam_rot: np.ndarray):
    # Set camera position and rotation
    camera = bpy.data.objects[camera_name]
    camera.location = cam_pos
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = cam_rot



def init_sun(sun_name: str, sun_pos: Vector, sun_rot: np.ndarray, light_energy: float):
    sun = bpy.data.objects[sun_name]
    sun.location = sun_pos
    sun.rotation_mode = 'QUATERNION'
    sun_quaternion_rot = sun_vectors_to_quaternions(np.array([sun_rot]))[0]
    sun.rotation_quaternion = sun_quaternion_rot
    lamp_data = bpy.data.lights[sun_name]
    lamp_data.energy = light_energy  # Set light energy


# main function
if __name__ == '__main__':
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    apply_blender_animation(motions_path=motions_path,
                            sun_path=sun_path,
                            info_path=info_path,
                            num_cols_per_object=dc.NUM_COLS_PER_OBJECT,
                            objects_dict=dc.object_names,
                            camera_name=dc.camera_name,
                            lightsource_name=dc.light_name,
                            cam_pos=dc.camera_position,
                            cam_rot=dc.camera_direction,
                            light_pos=dc.light_position,
                            light_rot=light_dir,
                            light_energy=dc.light_energy,
                            nb_im=nb_im)
