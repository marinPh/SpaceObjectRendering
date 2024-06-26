"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Renders multiple motions from the dataset. The script renders the
                first nb_im images from each motion. If nb_im is set to None, the
                script renders all images from each motion.
"""
import argparse

import bpy
import sys
import site
user_site_packages = site.getusersitepackages()
print (user_site_packages)
sys.path.append(user_site_packages)
import os
from tqdm import tqdm
parent_dir = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"--->{parent_dir}")

from render_setup import apply_blender_animation
import importlib

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("Utils"))


  # Reload the module to get the latest changes

import Utils.dataset_constants as dc

################################################
# User-defined inputs
proj_dir : str = os.path.dirname(os.path.dirname(__file__))
output_directory : str = os.path.join(proj_dir,"output")


# Input directory
input_directory: str = os.path.join(proj_dir,"input")
# Output directory


if len(sys.argv) < 3:
    print("Usage: python script.py arg1 arg2")
else:
    arg1= sys.argv[-2]
    arg2 = sys.argv[-1]
    print(f"Argument 1: {arg1}")
    print(f"Argument 2: {arg2}")
# Motion info
main_obj_name = arg1
pose_id = arg2

# Id of the objects to render
group_id: str = main_obj_name.split("_")[0]
# First motion id
first_motion_id: str = pose_id
# Number of motions to render
nb_motions: int = 1
# List of motions to render
motion_ids: list[str] = [f"{group_id}_{pose_id}"]
print(f"motion_ids={motion_ids}")

# Number of images to render (set to None for rendering all images from motion).
# Otherwise, renders the first nb_im images.
nb_im = None

# Should the script render the animation or simply apply it in Blender
# If there are more than one motion, only the last motion will appear in Blender
render_animation: bool = True

################################################
def log_render(mess):
    with open(os.path.join(output_directory, "log_render.txt"), "a") as log_file:
        log_file.write(f"{mess}\n")
def passIndex_correction(parent):
    parent.pass_index = 1
    for child in parent.children:
        child.pass_index = 1
        passIndex_correction(child)
def main() -> None:
    
    init_log_file(output_directory, motion_ids, dc.progress_log_file_name)
    
    for motion in motion_ids:
        print(motion)
        motions_path = os.path.join(input_directory, motion, dc.scene_gt_file_name)
        info_path = os.path.join(input_directory, motion, dc.scene_info_file_name)
        sun_path = os.path.join(input_directory, motion, dc.sun_orientations_file_name)
        if not os.path.exists(sun_path):
            log_render("no sunpath")
            sun_path = None
        log_render(f"sun_path={sun_path}")
        log_render(f"info_path={info_path}")
        log_render(f"num_cols={dc.NUM_COLS_PER_OBJECT}")
        log_render(f"object_name={dc.object_names}")
        log_render(f"camera_name={dc.camera_name}")
        log_render(f"light_name={dc.light_name}")
        log_render(f"camera_pos={dc.camera_position}")
        log_render(f"camera_dir={dc.camera_direction}")
        log_render(f"light_pos={dc.light_position}")
        log_render(f"light_default_direction={dc.light_default_direction}")
        #log_render(f"light_energy={dc.light_energy}")
        log_render(f"nb_im={nb_im}")
        
        sun_energy_path = os.path.join(input_directory, motion,"sun_energy.txt")
        with open(sun_energy_path, 'r') as file:
            sun_energies = [float(line.strip().split(',')[1]) for line in file.readlines()]
            print(sun_energies)
        print(f"-------->{motions_path}")
        nb_frames = apply_blender_animation(motions_path=motions_path, 
                                sun_path=sun_path,
                                info_path=info_path,
                                num_cols_per_object=dc.NUM_COLS_PER_OBJECT,
                                objects_dict=dc.object_names,
                                camera_name=dc.camera_name,
                                lightsource_name=dc.light_name,
                                cam_pos=dc.camera_position,
                                cam_rot=dc.camera_direction,
                                light_pos=dc.light_position,
                                light_rot=dc.light_default_direction,
                                sun_energies = sun_energies,
                                nb_im=nb_im)
        passIndex_correction(bpy.context.scene.objects[main_obj_name])
        
        # Get the nodes in the compositing tree
        nodes = bpy.context.scene.node_tree.nodes 
        if dc.mask_node_name not in nodes:
            raise KeyError("No mask node found in the compositing tree")
        # set nb of samples
        bpy.context.scene.cycles.samples = 2048
        print(f"-------->{bpy.context.scene.objects}")
        mask_node = nodes[dc.mask_node_name]
     
        #get the index of objects in the scene
  
        mask_node.base_path = os.path.join(output_directory, motion, dc.mask_folder_name)
        rgb_path = os.path.join(output_directory, motion, dc.render_folder_name) 
        
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        bpy.context.scene.render.filepath = rgb_path
        bpy.context.scene.frame_start = 0 
        bpy.context.scene.frame_current = 0 
        bpy.context.scene.frame_end = nb_frames - 1 
        
        # Register the handlers
        #creation and update of the progress bar with tqdm


        
# Initialize the progress bar
        pbar = tqdm(total=nb_frames)

        def update_progress_bar(scene):
            # Update the progress bar
            print("update_progress_bar")
            pbar.update(1)

        
        # Register the handlers
       
        #I want to see the object position in the scene
        
        
        
        
        

        # Render the animation
        #rendering without exiting blender
        bpy.ops.render.render( animation=True, write_still=False)
        bpy.ops.render.view_show()

        # Remove the handler when rendering is finished
        

        add_rendered_motion_to_log(output_directory, motion, dc.progress_log_file_name)
        
        
import blf

# Function to draw the progress bar

def init_log_file(output_directory : str, motion_ids : list[str], log_file_name : str) -> None:
    """Initializes the log file with the motion ids to render
    
    Args:
        output_directory (str): Output directory
        motion_ids (list[str]): List of motion ids
        log_file_name (str): Name of the log file
    """
    with open(os.path.join(output_directory, log_file_name), "w") as log_file:
        log_file.write("Motion ids to render:\n")
        for i, motion in enumerate(motion_ids):
            log_file.write(f"{motion} ")
            if i > 0 and i % 10 == 0:
                log_file.write("\n")
                
        log_file.write("\n\nRendered motions:\n")
        
def add_rendered_motion_to_log(output_directory : str, motion_id : str, log_file_name : str) -> None:
    """Adds the rendered motion to the log file
    
    Args:
        output_directory (str): Output directory
        motion_id (str): Motion id
        log_file_name (str): Name of the log file
    """
    with open(os.path.join(output_directory, log_file_name), "a") as log_file:
        log_file.write(f"{motion_id}\n")


if __name__ == "__main__":
    main()
