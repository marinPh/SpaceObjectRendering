"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: This script calculates the moment of inertia tensor of an object in Blender.
"""

print("looking for modules")
import argparse
import sys
import bpy
import numpy as np
import os
from mathutils import Vector
#import plotly
#import plotly.graph_objs as gFo
#import plotly.offline as pyo
from tqdm import tqdm
import bmesh
from mathutils.bvhtree import BVHTree

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
if len(sys.argv) < 2:
    print("Usage: python calculate_inertia_matrix.py <arg1>")
    sys.exit(1)

# Access the argument passed in the command line
arg1 = sys.argv[1]
print("Argument 1:", arg1)

main_obj_name = arg1.split("/")[3].split(".")[0]
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
    print(f"{bpy.context.view_layer.objects.active} is active object")
    print(f"{obj.children} the children of {obj.name}")

    log_inertia(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print(f"{obj.name} is active object")
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
        print(f"bbox:{i} {obj.dimensions} of {obj.name}")
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
def calculate_combined_inertia_matrix_and_approx_com(mesh_objects, total_mass, num_samples):
    I_local = np.zeros((3, 3))
    global_approx_com = np.zeros(3)

    all_inside_points = []

    min_corner, max_corner = combined_bbox(mesh_objects)

    bvh_trees = create_bvh_trees(create_bmesh_objects(mesh_objects))

    num_inside_points = 0

    print("\nSampling points...\n")
    print(f"min_corner: {min_corner}, max_corner: {max_corner}")

    with tqdm(total=num_samples) as progress_bar:
        while num_inside_points < num_samples:
            point = random_point_in_bbox(min_corner, max_corner)
            for obj in bvh_trees:
                if point_is_inside_tree(point, obj):
                    num_inside_points += 1
                    global_approx_com += point
                    all_inside_points.append(point)
                    colorful_progress_bar(num_inside_points, num_samples)
                    break
                
            progress_bar.update(num_inside_points - progress_bar.n)

    global_approx_com /= num_inside_points

    print("\nCalculating inertia matrix...\n")

    for point in all_inside_points:
        x, y, z = point - global_approx_com # If the center of mass is not the origin, we have to subtract the global_approx_com
        I_local[0, 0] += (y**2 + z**2)
        I_local[1, 1] += (x**2 + z**2)
        I_local[2, 2] += (x**2 + y**2)
        I_local[0, 1] -= x*y
        I_local[1, 0] -= x*y
        I_local[0, 2] -= x*z
        I_local[2, 0] -= x*z
        I_local[1, 2] -= y*z
        I_local[2, 1] -= y*z

    I_local *= total_mass / num_inside_points

    return I_local, global_approx_com, all_inside_points

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
"""def create_html_from_inside_points(all_inside_points, main_obj_name, output_directory):
    # Generate the 3D scatter plot
    x = [point[0] for point in all_inside_points]
    y = [point[1] for point in all_inside_points]
    z = [point[2] for point in all_inside_points]

    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='rgb(0, 0, 255)'))
    data = [trace]

    # Find the largest range among x, y, and z
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    range_z = max(z) - min(z)
    max_range = max(range_x, range_y, range_z)

     #Calculate aspect ratios for each axis
    aspect_ratio_x = range_x / max_range
    aspect_ratio_y = range_y / max_range
    aspect_ratio_z = range_z / max_range

    layout = go.Layout(title=f"Inside Points for '{main_obj_name}'",
                    scene=dict(xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z',
                                aspectmode='data',
                                aspectratio=dict(x=aspect_ratio_x,
                                                y=aspect_ratio_y,
                                                z=aspect_ratio_z)
                                )
                    )

    #fig = go.Figure(data=data, layout=layout)

    # Save the scatter plot to an HTML file
    #output_file_path_plot = os.path.join(output_directory, f"{main_obj_name}_inside_points.html")
    #pyo.plot(fig, filename=output_file_path_plot, auto_open=False)"""


# Get a reference to the main object
init_log_file(output_directory,log_file_name)
print("loading blend file")
bpy.ops.wm.open_mainfile(filepath=blend_file_path)



print("looking for main object")
print(f"{main_obj_name} this is the name")
print(bpy.data.objects.keys())

if main_obj := bpy.data.objects[main_obj_name]:
    print("main_object found")
    print(f"{type(main_obj)} is main object type")
    print(f"{type(bpy.context.view_layer.objects.active)} is active object type")

    # Make sure the object is at the origin, has no rotation and no scale
    main_obj.location = (0, 0, 0)
    main_obj.rotation_mode = 'QUATERNION'
    main_obj.rotation_quaternion = (1, 0, 0, 0)
    main_obj.scale = (1, 1, 1)

    # !!!! if you use scale an object in blender GUI, you have to apply the scale before you can use the script
    #main_obj.scale = (0.001, 0.001, 0.001)
    if main_obj.scale != (1, 1, 1):
        print(main_obj.scale)
        print("WARNING: The object has been scaled. Please apply the scale before running the script!\n")
        print(f"Scaling object '{main_obj_name}' by {main_obj.scale[0]}...\n")
        scale_all_children(main_obj.scale[0], main_obj)   
    # Apply all transformations and scales before continuing this is to avoid any issues when calculating the inertia matrix

    apply_all_transforms(main_obj)

    # Traverse hierarchy and find all mesh objects
    mesh_objects = traverse_hierarchy(main_obj, [])

    # Calculate the combined bounding box
    bbox = combined_bbox(mesh_objects)

    # Calculate the size of the object
    size = bbox[1] - bbox[0]  # size = max_corner - min_corner

    # Calculate the combined inertia matrix and approximate center of mass
    combined_inertia_matrix, global_approx_com, all_inside_points = calculate_combined_inertia_matrix_and_approx_com(mesh_objects, total_mass, num_samples)

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