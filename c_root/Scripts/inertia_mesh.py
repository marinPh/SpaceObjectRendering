import pymeshlab
import numpy as np
import os
import bpy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import save_info_to_files_utils as utils

#get from command line object_name

object_name = sys.argv[sys.argv.index("--") + 1]


#create parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

output_directory = os.path.join(parent_dir, "output","test")
object_path = os.path.join(parent_dir,"objects")

print(f"output_directory: {output_directory}")
print(f"object_path: {object_path}")


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
    
print(f"obj_path: {obj_path}")

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
print(f"csv_path: {csv_path}")

vertices = ms.current_mesh().vertex_matrix()
np.savetxt(os.path.join(csv_path,object_name+".csv"), vertices, delimiter=",", header="x,y,z", comments='')
#get the cloud of points into a numpy array
print(f"vertices: {vertices.shape}")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
utils.create_html_from_inside_points(vertices,object_name,output_directory)










