"""
Author:     Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Creates a sun object in the scene.
"""

import bpy

# Create sun object
lamp_data = bpy.data.lights.new(name="Lightsource", type='SUN') 
lamp_object = bpy.data.objects.new(name="Lightsource", object_data=lamp_data) 

# Link the lamp object to the Master Collection
bpy.context.scene.collection.objects.link(lamp_object) 