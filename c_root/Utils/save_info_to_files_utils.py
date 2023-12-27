"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Saves the camera info to a file.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import dataset_constants as dc


def create_html_from_inside_points(all_inside_points, main_obj_name, output_directory,pose_id):
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

    fig = go.Figure(data=data, layout=layout)

    #Save the scatter plot to an HTML file
    output_file_path_plot = os.path.join(output_directory, f"{main_obj_name}_inside_points.html")
    pyo.plot(fig, filename=output_file_path_plot, auto_open=False)

def save_camera_info_to_file(output_directory : str,
                             fov : int = dc.camera_fov, 
                             camera_position : np.ndarray = dc.camera_position, 
                             camera_orientation : np.ndarray = dc.camera_direction,
                             camera_info_file_name : str = dc.camera_info_file_name):
    with open(os.path.join(output_directory, camera_info_file_name), 'w') as f:
        f.write("--- Camera info ---\n")
        f.write(f"FOV: {fov}°\n")
        f.write(f"Camera position (m) [x, y, z]: [{', '.join([f'{value:.2f}' for value in camera_position])}]\n")
        f.write(f"Camera orientation (Quaternion) [w, x, y, z]): [{', '.join([f'{val:.2f}' for val in camera_orientation])}]\n")

   
def plot_segments(point1,point2):
    x = [point1[0],point2[0]]
    y = [point1[1],point2[1]]
    z = [point1[2],point2[2]]
    
   
    
    trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=2, color='rgb(255, 0, 0)'))
    return trace
def plot_square(point1,point2,point3,point4):
    x = [point1[0],point2[0],point3[0],point4[0],point1[0]]
    y = [point1[1],point2[1],point3[1],point4[1],point1[1]]
    z = [point1[2],point2[2],point3[2],point4[2],point1[2]]
    trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=2, color='rgb(255, 0, 0)'))
    return trace
def html_motion_points(all_inside_points, frustum_corners, main_obj_name, output_directory,pose_id):
    # Generate the 3D scatter plot
    x = [point[0] for point in all_inside_points]
    y = [point[1] for point in all_inside_points]
    z = [point[2] for point in all_inside_points]
    origin_trace = go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=4, color='rgb(255, 0, 255)'))
    end_trace = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', marker=dict(size=4, color='rgb(0, 255, 255)'))
    
    # Find the largest range among x, y, and z
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    range_z = max(z) - min(z)
    max_range = max(range_x, range_y, range_z)
    

    # Calculate aspect ratios for each axis
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

    

    # Cr x_frustum = frustum_corners[:, 0]
  
   

    # Create line segments for the frustum
    line_segments = []
   
    
    apex_square = plot_square(frustum_corners[0],frustum_corners[1],frustum_corners[2],frustum_corners[3])
    base_square = plot_square(frustum_corners[4],frustum_corners[5],frustum_corners[6],frustum_corners[7])
    
    for i in range(4):
        line_segments.append(plot_segments(frustum_corners[i], frustum_corners[(i + 4)]))
        
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='rgb(0, 0, 255)'))
    
    data = [trace, apex_square, base_square, origin_trace, end_trace]
    
    for i in line_segments:
        data.append(i)
    

    fig = go.Figure(data=data, layout=layout)

    # Save the scatter plot to an HTML file
    output_file_path_plot = os.path.join(output_directory, f"{main_obj_name}_{pose_id}_motion_view.html")
    fig.write_html(output_file_path_plot, auto_open=False)