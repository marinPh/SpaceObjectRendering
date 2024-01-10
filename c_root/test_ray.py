import argparse
import sys
import bpy
import numpy as np
import os
from mathutils import Vector
import cProfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

import plotly.graph_objects as go
#import plotly
#import plotly.graph_objs as gFo
#import plotly.offline as pyo
from tqdm import tqdm
import bmesh
from mathutils.bvhtree import BVHTree
def create_bvh_trees(bmesh_objects):
    return [BVHTree.FromBMesh(bm) for bm in bmesh_objects]

def create_bmesh_objects(mesh_objects):
    
    bmesh_objects = []
    for obj in mesh_objects:
        mesh = obj.data
        matrix = obj.matrix_world
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(matrix)
        bmesh_objects.append(bm)
    return bmesh_objects

def traverse_hierarchy(obj, mesh_objects):
    
    if obj.type == 'MESH':
        mesh_objects.append(obj)
    for child in obj.children:
        traverse_hierarchy(child, mesh_objects)
    return mesh_objects
def point_is_inside_tree(points, bvh_trees):
   
   #TODO vectorize this function with numpy
    num_points = len(points)
    num_rays = 7
    #I want only the first 10 points


    # Generate all random directions at once
    
    ray_directions = np.random.uniform(-1, 1, size=(num_points, num_rays, 3))
    #for each point, have an array that has each hit point of the ray for each bvh tree
    #using the function ray_cast_func
   
    intersections = np.array([[ray_cast_func(point, ray_directions[i], bvh_tree) for bvh_tree in bvh_trees] for i, point in enumerate(points)])
    print(f"intersections shape: {intersections.shape}")
    bool_intersections = np.sum(intersections,axis=1) % 2
    print (f"bool_intersections: {bool_intersections.shape}")
  
  
    return bool_intersections
   
"""
    is_inside = True
    for bvh_tree in bvh_trees:
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
            if is_inside:
                return is_inside
            
    return is_inside"""
    
def ray_cast_func(point,ray_direction,bvh_tree)->bool:
    point.reshape(-1,3)
    ray_direction.reshape(-1,3)
    default_value = np.array([None, None, None])  # use None as default value
    
   
    hits = np.array([bvh_tree.ray_cast(point,ray)[1] if bvh_tree.ray_cast(point,ray)[0] is True else default_value for ray in ray_direction])
    #use the function check_hit to check if the point is inside the object
    print(f"hits: {hits}")
    intersections = np.array([check_hit(hit,ray_direction[i],bvh_tree) for i,hit in enumerate(hits)])

    #print(f"intersections: {intersections}, {point}")
    # return true if one of the rays intersect the object an odd number of times 
  
    return np.sum(intersections,axis=0) % 2 == 1

def check_hit(hit,ray_direction,bvh_tree):  
  
    if hit is None or hit[0] is None:
        return 0
    intersection = 1
    bo = True
    while hit is not None and bo :
        print(f"hit: {hit} + {Vector(ray_direction)*1e-4 } = {intersection}")
        bo,hit,_,_ = bvh_tree.ray_cast(hit + Vector(ray_direction)*1e-4, ray_direction)
       
      
       
        intersection += 1
    return intersection 

def scale_all_children(scale_factor: float, object):
    """Scale all children of an object

    Args:
        scale_factor (float): Scale factor
        object (bpy.types.Object): Object
    """
    for child in object.children:
        print(f"child: {child.name} scaled")
        child.scale = (scale_factor, scale_factor, scale_factor)
        print(object.scale)
        scale_all_children(scale_factor, child)


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

# Function that calculates the aggregate inertia matrix and approximate center of mass of a system of mesh objects
def calculate_combined_inertia_matrix_and_approx_com(mesh_objects, total_mass, num_samples):
    I_local = np.zeros((3, 3))
    global_approx_com = np.zeros(3)

    all_inside_points = []

    min_corner, max_corner = mesh_objects[0].bound_box[0], mesh_objects[0].bound_box[6]

    bvh_trees = bpy.data.objects['Sphere'].bvhtree

    num_inside_points = 0

    print("\nSampling points...\n")
    print(f"min_corner: {min_corner}, max_corner: {max_corner}")


    #optimization of the above code with numpy arrays
    # the goal is having num_inside_points to be  equal to num_samples
    # and all_inside_points to be a list of points inside the object
    # and global_approx_com to be the sum of all the points inside the object
    # array_of_points is a numpy array of shape (num_samples,3) with random points inside the bounding box
    
    #checkin with numpy arrays if the points are inside the object
    array_of_points_inside = np.array([]).reshape(-1,3)
    with tqdm(total=num_samples) as progress_bar:
        
        while array_of_points_inside.shape[0] < num_samples:
            #print(1)
            array_of_points = np.random.uniform(min_corner, max_corner, ((num_samples-array_of_points_inside.shape[0]),3))
            print(f"array_of_points shape: {array_of_points.shape}")

           
            #array of booleans of shape (num_samples) with True if the point is inside the object that is in the bvh tree
            array_of_booleans = point_is_inside_tree(array_of_points, bvh_trees)
      
            print(f"array_of_booleans average: {np.average(array_of_booleans)}")
       
            print(f"array_of_booleans shape: {array_of_booleans.shape}")
            #array of points inside the object
            print(f"array_of_points_inside shape: {array_of_points_inside.shape}")
            
            #keep only the points inside the object
            

            array_of_points_inside = np.append(array_of_points_inside,array_of_points[array_of_booleans == 1],axis=0)
            
            print(f"array_of_points_inside shape: {array_of_points_inside.shape}")
            #number of points inside the object
        
            #sum of the points inside the object
            progress_bar.update(array_of_points_inside.shape[0]-progress_bar.n)
            global_approx_com = np.sum(array_of_points_inside, axis=0)
            
            
            #list of points inside the object
            
            
            #update the progress bar
            
            #break the for loop
        
            #update the array of points
            




    num_inside_points = array_of_points_inside.shape[0]
    all_inside_points = array_of_points_inside

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
    print(num_inside_points)
    I_local *= total_mass / num_inside_points
   
    return I_local, global_approx_com, all_inside_points

#we want to test on y sphere,

test_input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "objects","blend","6_CHEOPS_LP.blend")
bpy.ops.wm.open_mainfile(filepath=test_input_path)
main_obj = bpy.data.objects['6_CHEOPS_LP']
print(f"main_obj: {main_obj.scale}")
scale_all_children(main_obj.scale[0], main_obj)

bm =traverse_hierarchy(main_obj, [])


bvh_trees = create_bvh_trees(create_bmesh_objects(bm))


bbox =main_obj.bound_box 
#print first object's position
print(f"position: {main_obj.location}")

bbox = combined_bbox(bm)
bbox_center = (bbox[0] + bbox[1]) / 2
print(f"bbox_center: {bbox_center}")
print(f"bbox: {bbox}")
# Set the number of samples to use for the Monte Carlo approximation
origin = bbox_center
ray_direction = np.array([[1,1,1]]).astype(np.float32)
cast = np.array([ray_cast_func(origin,ray_direction,bvh_tree) for  bvh_tree in bvh_trees])
print(f"bbox: {bbox}")
print(f"main_obj: {main_obj}")
print(f"cast: {cast}")
