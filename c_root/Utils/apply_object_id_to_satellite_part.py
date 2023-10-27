"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Applies a Pass Index to all selected objects.
"""

import bpy

pass_index = 4  # Replace this value with the desired Pass Index for the selected objects

# This script will apply the Pass Index to all selected objects

for obj in bpy.context.selected_objects: # type: ignore
    obj.pass_index = pass_index
