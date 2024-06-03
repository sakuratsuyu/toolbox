# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import os
import json
import bpy
import numpy as np
import math

DEBUG = False
            
VIEWS = 50
RESOLUTION = 800
RESULTS_PATH = f'images_{VIEWS:03d}'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
UPPER_VIEWS = True


scene = bpy.context.scene
save_path = bpy.path.abspath(f"//{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Render Optimizations
scene.render.use_persistent_data = True

# Set up rendering of depth map.
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
scene.render.image_settings.file_format = str(FORMAT)
scene.render.image_settings.color_depth = str(COLOR_DEPTH)
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
# Background
scene.render.dither_intensity = 0.0
scene.render.film_transparent = True

gbuffer_output = {}

if not DEBUG:
    if tree.nodes.find('Script Generated Render Layers Roughness') == -1:
        # Create input render layer node.
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers.name = 'Script Generated Render Layers Roughness'

        roughness_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        roughness_file_output.name = 'roughness_output'
        links.new(render_layers.outputs['DiffCol'], roughness_file_output.inputs[0])
        gbuffer_output['roughness'] = roughness_file_output
    else:
        roughness_file_output = tree.nodes['diffuse_output']
        gbuffer_output['roughness'] = roughness_file_output


# Create collection for objects not to render with background

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


# Set up camera
cam = scene.objects['Camera']
cam.location = (0, 6, 0)
if cam.constraints.find("Track To") == -1:
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
else:
    cam_constraint = cam.constraints["Track To"]
    b_empty = cam_constraint.target


stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in gbuffer_output.values():
        output_node.base_path = save_path

with open(save_path + '/' + 'transforms.json', 'r') as f:
    file = json.load(f)
    poses = file['frames']

for i in range(len(poses)):
    os.makedirs(os.path.join(save_path, f'{i:03d}'), exist_ok=True)
    scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}')

    rot = poses[i]['rotation']
    print(rot)
    b_empty.rotation_euler = rot

    for output_node in gbuffer_output.values():
        output_node.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + output_node.name + "_")

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still