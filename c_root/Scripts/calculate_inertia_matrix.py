"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: This script calculates the moment of inertia tensor of an object in Blender.
"""

print("looking for modules")
import argparse
import sys
import bpy
import site
user_site_packages = site.getusersitepackages()
print (user_site_packages)
sys.path.append(user_site_packages)
import numpy as np
import os
from mathutils import Vector
import pymeshlab
#import plotly
#import plotly.graph_objs as gFo
#import plotly.offline as pyo
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from Utils import save_info_to_files_utils as utils

################################################
# User-defined inputs

def scale_all_children(scale_factor: float, object):
    """Scales all children of an object by a given factor

    Args:
        scale_factor (float): Scale factor
        object (bpy.types.Object): Object
    """
    for child in object.children:
        child.scale = (scale_factor, scale_factor, scale_factor)
        scale_all_children(scale_factor, child)
    

# Name of the main object
arg1 = sys.argv[sys.argv.index("--") + 1]


main_obj_name = arg1
# Total mass of the entire system (kg)


total_mass = 12200
# Number of random points to sample
num_samples = 100000
# Output directory for the inertia matrix text file
log_file_name = "log_inertia.txt"
proj_dir : str = os.path.dirname(os.path.dirname(__file__))
input_directory : str = os.path.join(proj_dir,"input")
output_directory : str = os.path.join(proj_dir,"objects","inertia")
blend_file_path = os.path.join(proj_dir,"objects","blend",f"{main_obj_name}.blend")

# Name of the output text file for the moment of inertia tensor
# This file will be used as input for the diagonalization script
output_file_name = main_obj_name+'_inertia_matrix.txt'
# Name of the output text file for other information (e.g. object name, total mass, bounding box, etc.)
# The name of this file will be prepended with the object name
output_info_file_name = 'info.txt'

################################################

# Function that applies all transforms to an object and its children


def colorful_progress_bar(current_iteration, total_iterations, bar_length=40):
    percent = current_iteration / total_iterations
    arrow = "=" * int(round(bar_length * percent))
    spaces = " " * (bar_length - len(arrow))

    # Define colors using ANSI escape codes
    color_start = "\033[1;32m"  # Green text
    color_end = "\033[0m"  # Reset to default color

    progress_bar = f"{color_start}[{arrow}{spaces}] {int(percent * 100)}%{color_end}"
    return progress_bar






def log_inertia(mess,file_name = log_file_name):
    with open(os.path.join(proj_dir,"logs", file_name), "a") as log_file:
        log_file.write(f"{mess}\n")


def init_log_file(log_dir: str, log_file_name: str) -> None:
    """Initializes the log file with the motion ids to render

    Args:
        output_directory (str): Output directory
        motion_ids (list[str]): List of motion ids
        log_file_name (str): Name of the log file
    """
    with open(os.path.join(log_dir, log_file_name), "w") as log_file:
        log_file.write("\n\ninit file:\n")

init_log_file(os.path.join(proj_dir,"logs"),log_file_name)
def apply_all_transforms(obj):
    log_inertia("apply_all_trans")
   

    log_inertia(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
   
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) 
    obj.select_set(False)
    for child in obj.children:

        apply_all_transforms(child)

# Function to traverse object hierarchy and find all mesh objects
def traverse_hierarchy(obj, mesh_objects):
    log_inertia("traverse")
    if obj.type == 'MESH':
        mesh_objects.append(obj)
    for child in obj.children:
        traverse_hierarchy(child, mesh_objects)
    return mesh_objects

# Function to create bmesh objects from mesh objects
def create_bmesh_objects(mesh_objects):
    log_inertia("create_bmesh_objects")
    bmesh_objects = []
    for obj in mesh_objects:
        mesh = obj.data
        matrix = obj.matrix_world
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(matrix)
        bmesh_objects.append(bm)
    return bmesh_objects

# Function to create BVH trees from bmesh objects
def create_bvh_trees(bmesh_objects):
    return [BVHTree.FromBMesh(bm) for bm in bmesh_objects]

# Function to generate a random point inside the bounding box
def random_point_in_bbox(min_corner, max_corner):
    point = np.zeros(3)
    for i in range(3):
        point[i] = np.random.uniform(min_corner[i], max_corner[i])
    return point

# Function to calculate the combined bounding box of multiple mesh objects
def combined_bbox(mesh_objects):
    min_corner = np.zeros(3)
    max_corner = np.zeros(3)
    for i, obj in enumerate(mesh_objects):
        bbox = [Vector(corner) for corner in obj.bound_box]
        
        if i == 0:
            min_corner = np.array(
                [
                    min(p.x for p in bbox),
                    min(p.y for p in bbox),
                    min(p.z for p in bbox),
                ]
            )
            max_corner = np.array(
                [
                    max(p.x for p in bbox),
                    max(p.y for p in bbox),
                    max(p.z for p in bbox),
                ]
            )
        else:
            min_corner = np.minimum(
                min_corner,
                [
                    min(p.x for p in bbox),
                    min(p.y for p in bbox),
                    min(p.z for p in bbox),
                ],
            )
            max_corner = np.maximum(
                max_corner,
                [
                    max(p.x for p in bbox),
                    max(p.y for p in bbox),
                    max(p.z for p in bbox),
                ],
            )
    return min_corner, max_corner

# Function that checks if a point is inside a BVH tree
def point_is_inside_tree(point, bvh_tree):
   
    is_inside = True
    for _ in range(7):
        point = Vector(point)
        ray_direction = Vector([np.random.uniform(-1, 1) for _ in range(3)])

        # Cast a ray from the point in a random direction and count intersections
        intersections = 0
        hit, _, _, _ = bvh_tree.ray_cast(point, ray_direction)
        while hit is not None:
            intersections += 1
            hit, _, _, _ = bvh_tree.ray_cast(hit + ray_direction * 1e-4, ray_direction)
           

        # If the ray intersects the object an odd number of times, the point is inside the object
        is_inside = is_inside and intersections % 2 == 1
       

    return is_inside

# Function that calculates the aggregate inertia matrix and approximate center of mass of a system of mesh objects
def calculate_combined_inertia_matrix_and_approx_com(mesh_objects, total_mass, num_samples,object_name):
    I_local = np.zeros((3, 3))
    global_approx_com = np.zeros(3)

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

    output_directory = os.path.join(parent_dir, "output","test")
    object_path = os.path.join(parent_dir,"objects")

  

    #get the object from blend file
    file = bpy.ops.wm.open_mainfile(filepath=os.path.join(object_path,"blend",object_name+".blend"))
    #export in obj format
    #select object to export
    bpy.ops.object.select_all(action='DESELECT')
    #select object and its hierarchy
    bpy.data.objects[object_name].select_set(True)
    #set object as active
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
    bpy.ops.object.select_hierarchy(direction='CHILD', extend=True)


    #export
    obj_path = os.path.join(object_path,"obj")
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)



    bpy.ops.export_scene.obj(filepath =os.path.join(obj_path,f"{object_name}.obj"), use_selection=True)

    #create mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(obj_path,f"{object_name}.obj"))
    #create cloud of points of mesh using montecarlo sampling
    ms.apply_filter('generate_sampling_montecarlo', samplenum=100000)
    #save cloud of points in csv file
    csv_path = os.path.join(object_path,"csv")
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    vertices = ms.current_mesh().vertex_matrix()
    np.savetxt(os.path.join(csv_path,object_name+".csv"), vertices, delimiter=",", header="x,y,z", comments='')
    num_inside_points = vertices.shape[0]
#get the cloud of points into a numpy array

    global_approx_com = np.mean(vertices, axis=0)
    html_path = os.path.join(object_path,"html")
    if not os.path.exists(html_path):
        os.mkdir(html_path)
    utils.create_html_from_inside_points(vertices,object_name,html_path)

    print("\nCalculating inertia matrix...\n")

    shifted_vertices = vertices - global_approx_com

    # Extract individual coordinates
    x, y, z = shifted_vertices[:, 0], shifted_vertices[:, 1], shifted_vertices[:, 2]
   
    # Calculate the contributions to the inertia matrix
    I_local = np.zeros((3, 3))
    I_local[0, 0] = np.sum(y**2 + z**2)
    I_local[1, 1] = np.sum(x**2 + z**2)
    I_local[2, 2] = np.sum(x**2 + y**2)
    I_local[0, 1] = I_local[1, 0] = -np.sum(x*y)
    I_local[0, 2] = I_local[2, 0] = -np.sum(x*z)
    I_local[1, 2] = I_local[2, 1] = -np.sum(y*z)

    I_local *= total_mass / num_inside_points

    return I_local, global_approx_com, vertices

# Function that prints a matrix in a nice format
def print_formatted_matrix(matrix):
    formatted_matrix = "[\n"
    for row in matrix:
        formatted_matrix += "    [" + ', '.join(f"{x: .2f}" for x in row) + "],\n"
    formatted_matrix = formatted_matrix[:-2] + "\n]"
    return formatted_matrix

# Function that saves a matrix to a file
def save_matrix_to_file(file_path, matrix, main_obj_name):
    with open(file_path, 'w') as f:
        f.write(f"Total combined inertia matrix for object '{main_obj_name}':\n")
        formatted_matrix = "np.array([\n"
        for row in matrix:
            formatted_matrix += "    [" + ', '.join(f"{x: .2f}" for x in row) + "],\n"
        formatted_matrix = f"{formatted_matrix[:-2]}])"
        f.write(formatted_matrix)

def save_info_to_file(file_path, bbox, size, mass, num_samples, global_approx_com):
    with open(file_path, 'w') as f:
        f.write(f"Information for object '{main_obj_name}':\n")
        f.write(f"\nMass: {mass} kg\n")
        f.write(f"Dimensions (along x, y, z): {size[0]:.2f} m, {size[1]:.2f} m, {size[2]:.2f} m\n")  # Adjusted line
        f.write(f"Center of Mass (x, y, z): [{global_approx_com[0]:.2f}, {global_approx_com[1]:.2f}, {global_approx_com[2]:.2f}]\n")  # New line
        f.write(f"\nBounding box:\n")
        f.write(f"    Min corner (x, y, z): [{bbox[0][0]:.4f}, {bbox[0][1]:.4f}, {bbox[0][2]:.4f}]\n")
        f.write(f"    Max corner (x, y, z): [{bbox[1][0]:.4f}, {bbox[1][1]:.4f}, {bbox[1][2]:.4f}]\n")
        f.write(f"\nNumber of samples for inertia matrix: {num_samples}\n")

# Function that creates an HTML file with a 3D scatter plot of the points inside the object


# Get a reference to the main object
init_log_file(output_directory,log_file_name)
print("loading blend file")
bpy.ops.wm.open_mainfile(filepath=blend_file_path)



print("looking for main object")



if main_obj := bpy.data.objects[main_obj_name]:
    print("main_object found")
  

    # Make sure the object is at the origin, has no rotation and no scale
    main_obj.location = (0, 0, 0)
    main_obj.rotation_mode = 'QUATERNION'
    main_obj.rotation_quaternion = (1, 0, 0, 0)
    

    # !!!! if you use scale an object in blender GUI, you have to apply the scale before you can use the script
    #main_obj.scale = (0.001, 0.001, 0.001)
    if main_obj.scale != (1, 1, 1):
        print(f"main_obj.scale: {main_obj.scale}")
        print("WARNING: The object has been scaled. Please apply the scale before running the script!\n")
        print(f"Scaling object '{main_obj_name}' by {main_obj.scale[0]}...\n")
        #scale_all_children(main_obj.scale[0], main_obj)   
    # Apply all transformations and scales before continuing this is to avoid any issues when calculating the inertia matrix
    #main_obj.scale = (1, 1, 1)
    apply_all_transforms(main_obj)

    # Traverse hierarchy and find all mesh objects
    mesh_objects = traverse_hierarchy(main_obj, [])

    # Calculate the combined bounding box
    bbox = combined_bbox(mesh_objects)

    # Calculate the size of the object
    size = bbox[1] - bbox[0]  # size = max_corner - min_corner

    # Calculate the combined inertia matrix and approximate center of mass
    combined_inertia_matrix, global_approx_com, all_inside_points = calculate_combined_inertia_matrix_and_approx_com(mesh_objects, total_mass, num_samples,main_obj_name)

    # Print the results
    print("Global approximate center of mass: [{:.4f}, {:.4f}, {:.4f}]\n".format(global_approx_com[0], global_approx_com[1], global_approx_com[2]))
    # Check if the global approximate center of mass is almost at the origin (within 0.01)
    if abs(global_approx_com[0]) > 0.01 or abs(global_approx_com[1]) > 0.01 or abs(global_approx_com[2]) > 0.01:
        print("WARNING: The origin is not at the calculated center of mass. Please change it for correct motions!\n")
    print(
        f"Total combined inertia matrix for object '{main_obj_name}': \n{print_formatted_matrix(combined_inertia_matrix)}\n"
    )

    # Print the bounding box and the size
    print(f"Information for object '{main_obj_name}':\n")
    print(f"Mass: {total_mass} kg")
    print(f"Dimensions (along x, y, z): {size[0]:.2f} m, {size[1]:.2f} m, {size[2]:.2f} m\n")  # Adjusted line
    print(f"Center of Mass: [{global_approx_com[0]:.2f}, {global_approx_com[1]:.2f}, {global_approx_com[2]:.2f}]\n")  # New line
    print("Bounding box:")
    print(f"    Min corner (x, y, z): [{bbox[0][0]:.2f}, {bbox[0][1]:.2f}, {bbox[0][2]:.2f}]")
    print(f"    Max corner (x, y, z): [{bbox[1][0]:.2f}, {bbox[1][1]:.2f}, {bbox[1][2]:.2f}]\n")
    print(f"Number of samples for inertia matrix: {num_samples}\n")

    print("Saving results to files...")

    # Save the total inertia matrix to a text file
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_path_combined = os.path.join(output_directory, output_file_name)
    save_matrix_to_file(output_file_path_combined, combined_inertia_matrix, main_obj_name)

    # Create an HTML file with a 3D scatter plot of all the inside points
    #create_html_from_inside_points(all_inside_points, main_obj_name, output_directory)

    # Save the bounding box and the size to a file
    output_file_name_info = f'{main_obj_name}_{output_info_file_name}'
    output_file_path_info = os.path.join(output_directory, output_file_name_info)
    save_info_to_file(output_file_path_info, bbox, size, total_mass, num_samples, global_approx_com)
else:
    print("hein??")
    print("Done!\n")