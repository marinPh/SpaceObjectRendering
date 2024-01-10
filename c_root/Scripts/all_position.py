import sys
import os
import numpy as np
import re

# Add the path of the folder where the module is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Utils import dataset_constants as dc
from Utils import save_info_to_files_utils as utils
test_nb = 0
folder_tergets = range(1 + test_nb*50,1 + (test_nb+1)*50)

all_position = np.array([]).reshape(-1,3)
tot_frames = 0
proc =0
objec_name ="6_CHEOPS_LP"
object_id = objec_name.split("_")[0]
scrap_dir = os.path.join(parent_dir, 'input')
#we want all folders in the scrap_dir that respect this format: 
# {object_id}_{number} with any number
folders = os.listdir(scrap_dir)
print(folders)
print(len(folders))
for folder in folders:
    
    
    #we want all files in the folder that respect named scene_gt.txt 
    #scene_gt.txt contains the 3D coordinates of the object in the scene is a csv format file
    print(folder)
    if folder.split("_")[1] not in [str(i) for i in folder_tergets]:
        continue
    
    proc+=1
    scene_gt = np.loadtxt(os.path.join(scrap_dir, folder, 'scene_gt.txt'), delimiter=',')
    
  
    
    #position of the object are the last 3 columns of scene_gt.txt
    
    position = scene_gt[:, -3:]
    
    all_position = np.vstack((all_position, position))
    
print (f"all_position shape: {all_position.shape}")
print (f"mean position: {np.mean(all_position, axis=0)}")
print (f"std position: {np.std(all_position, axis=0)}")
print (f"mean nb of frames: {all_position.shape[0]/proc}")
print (f"proc = {proc   }")

size = dc.object_size(objec_name)

frustum_corners = dc.create_square_fov(dc.calculate_distance_with_fov(size,dc.max_coverage_ratio))
frustum_corners = np.vstack((frustum_corners, dc.create_square_fov(dc.calculate_distance_with_fov(size,dc.min_coverage_ratio))))
if not os.path.exists(os.path.join(parent_dir,"testing")):
    os.makedirs(os.path.join(parent_dir,"testing"))
utils.all_points_plot(all_position, frustum_corners,object_id, os.path.join(parent_dir,"testing"))
utils.visualising_distr(np.mean(all_position, axis=0),np.diag(np.std(all_position,axis=0)),frustum_corners,f"{objec_name}_{test_nb}_motion_dirtribution")
mean, cov = dc.mean_and_covariance_matrix(objec_name)
utils.visualising_distr(mean,cov,frustum_corners,f"{objec_name}_{test_nb}_target_point_distribution")
print(f"output saved in {os.path.join(parent_dir,'testing')}")
    
    
    