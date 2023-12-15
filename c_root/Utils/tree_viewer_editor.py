import bpy
import sys
import os


#get name from command line
object_name = sys.argv[sys.argv.index("--") + 1]

# Create parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
blend_file_path = os.path.join(parent_dir,"objects","blend")
#load blend file
file = bpy.ops.wm.open_mainfile(filepath=os.path.join(blend_file_path,object_name+".blend"))
#look into composition
bpy.context.scene.use_nodes = True
nodes = bpy.context.scene.node_tree.nodes
#look for the mask node
print(f"nodes: {nodes.keys()}")
mask_node = nodes["mask_output"]
#and the node that is connected to it
input_node = mask_node.inputs[0].links[0].from_node
print(f"input_node: {input_node.name}")
#if node is not invert color, create one
if input_node.name != 'Invert Color':
    invert_node = nodes.new("CompositorNodeInvert")
    print(f"invert_node: {invert_node.name}")
    #set it between the input node and the mask node
    # Link the new node to the existing nodes
    scene = bpy.context.scene
    compositor = scene.node_tree
    print(f"invert_node inputs: {invert_node.inputs.keys()}")
    print(f"invert_node outputs: {invert_node.outputs.keys()}")
    print(f"input_node outputs: {input_node.outputs.keys()}")
    print(f"mask_node inputs: {mask_node.inputs.keys()}")
    link1 = compositor.links.new(input_node.outputs['Value'], invert_node.inputs['Color'])
    link2 = compositor.links.new(invert_node.outputs['Color'], mask_node.inputs["Image"])
    

    
    
#save blend file
bpy.ops.wm.save_mainfile(filepath=os.path.join(blend_file_path,object_name+".blend"))






