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
RANDOM_VIEWS = False
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
    if tree.nodes.find('Script Generated Render Layers') == -1:
        # Create input render layer node.
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        render_layers.name = 'Script Generated Render Layers'

        alpha_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        alpha_file_output.name = 'alpha_output'
        links.new(render_layers.outputs['Alpha'], alpha_file_output.inputs[0])
        gbuffer_output['alpha'] = alpha_file_output

        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.name = 'normal_output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        gbuffer_output['normal'] = normal_file_output

        diffuse_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        diffuse_file_output.name = 'diffuse_output'
        links.new(render_layers.outputs['DiffCol'], diffuse_file_output.inputs[0])
        gbuffer_output['diffuse'] = diffuse_file_output
    else:
        alpha_file_output = tree.nodes['alpha_output']
        gbuffer_output['alpha'] = alpha_file_output
        normal_file_output = tree.nodes['normal_output']
        gbuffer_output['normal'] = normal_file_output
        diffuse_file_output = tree.nodes['diffuse_output']
        gbuffer_output['diffuse'] = diffuse_file_output




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
cam.location = (0, 6.0, 0.0)
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

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'frames': []
}

if not DEBUG:
    for output_node in gbuffer_output.values():
        output_node.base_path = save_path

for i in range(0, VIEWS):
    os.makedirs(os.path.join(save_path, f'{i:03d}'), exist_ok=True)
    if RANDOM_VIEWS:
        scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}')
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
            b_empty.rotation_euler = rot
        else:
            rot = np.random.uniform(0, 2 * np.pi, size=3)
            b_empty.rotation_euler = rot
    else:
        print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
        # scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{int(i * stepsize):03d}')
        scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}')
        b_empty.rotation_euler[2] += math.radians(stepsize)
        rot = b_empty.rotation_euler

    for output_node in gbuffer_output.values():
        output_node.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + output_node.name + "_")

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': os.path.relpath(scene.render.filepath, start=save_path),
        'rotation': [rot[0], rot[1], rot[2]],
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(save_path + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
