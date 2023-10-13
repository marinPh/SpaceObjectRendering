"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Renders multiple motions from the dataset. The script renders the
                first nb_im images from each motion. If nb_im is set to None, the
                script renders all images from each motion.
"""

import bpy
import sys
import os
import importlib

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("Utils"))

import apply_motion_for_render_function
  # Reload the module to get the latest changes
from apply_motion_for_render_function import apply_blender_animation
import Utils.dataset_constants as dc

################################################
# User-defined inputs
proj_dir : str = "C:\\Users\\marin\\Documents\\BA5\\ProjB\\hubble" 
output_directory : str = os.path.join(proj_dir,"output")

# Input directory
input_directory: str = os.path.join(proj_dir,"input")
# Output directory
output_directory: str = input_directory

# Id of the objects to render
group_id: str = "02"
# First motion id
first_motion_id: str = "000"
# Number of motions to render
nb_motions: int = 1
# List of motions to render
motion_ids: list[str] = [f"{group_id}{str(i + int(first_motion_id)).zfill(3)}" for i in range(nb_motions)]

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
def main() -> None:
    
    init_log_file(output_directory, motion_ids, dc.progress_log_file_name)
    
    for motion in motion_ids:
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
        log_render(f"light_energy={dc.light_energy}")
        log_render(f"nb_im={nb_im}")
        

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
                                light_energy=dc.light_energy,
                                nb_im=nb_im)
        
        # Get the nodes in the compositing tree
        nodes = bpy.context.scene.node_tree.nodes 
        if dc.mask_node_name not in nodes or dc.seg_node_name not in nodes:
            raise KeyError("No mask or segmentation node found in the compositing tree")
        
        mask_node = nodes[dc.mask_node_name]
        seg_node = nodes[dc.seg_node_name]
        
        mask_node.base_path = os.path.join(output_directory, motion, dc.mask_folder_name)
        seg_node.base_path = os.path.join(output_directory, motion, dc.segmentation_folder_name)
        bpy.context.scene.render.filepath = os.path.join(output_directory, motion, dc.render_folder_name) 
        
        bpy.context.scene.frame_start = 0 
        bpy.context.scene.frame_current = 0 
        bpy.context.scene.frame_end = nb_frames - 1 
        
        # Register the handlers
        bpy.app.handlers.render_post.append(update_progress_bar)

# Render the animation
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)

# Remove the handler when rendering is finished
        bpy.app.handlers.render_post.remove(update_progress_bar)
        add_rendered_motion_to_log(output_directory, motion, dc.progress_log_file_name)
        
        
import blf

# Function to draw the progress bar
def draw_progress_bar():
    progress = bpy.context.scene.frame_current / bpy.context.scene.frame_end
    width = 300
    height = 20
    x = (bpy.context.area.width - width) / 2
    y = 50
    
    # Clear the area for the progress bar
    bpy.context.area.regions[5].tag_redraw()
    
    # Draw the progress bar
    blf.position(0, x, y, 0)
    blf.size(0, 20, 72)
    blf.draw(0, f"Rendering Frame {bpy.context.scene.frame_current}/{bpy.context.scene.frame_end}")
    bpy.context.area.regions[5].tag_redraw()
    
# Function to update the progress bar
def update_progress_bar(scene):
    if bpy.ops.render.is_running:
        draw_progress_bar()    
    
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
