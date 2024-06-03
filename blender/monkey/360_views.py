# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import os
import json
import bpy
import numpy as np
import math

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

DEBUG = False

VIEWS = 100
RESOLUTION = 800
RESULTS_PATH = f'images_{VIEWS:03d}'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = False
UPPER_VIEWS = True

scene = bpy.context.scene
save_path = bpy.path.abspath(f"//{RESULTS_PATH}")

stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

os.makedirs(save_path, exist_ok=True)

# Render Optimizations
scene.render.use_persistent_data = True

# Set up rendering of depth map.
scene.use_nodes = True
tree = scene.node_tree
links = tree.links
nodes = tree.nodes

# Add passes for additionally dumping albedo and normals.
scene.render.image_settings.file_format = str(FORMAT)
scene.render.image_settings.color_depth = str(COLOR_DEPTH)
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
# Background
scene.render.dither_intensity = 0.0
scene.render.film_transparent = True



# Set up camera
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty

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


# Set up shader node tree
material = bpy.data.materials[0]
shader_tree = material.node_tree
shader_links = shader_tree.links
shader_nodes = shader_tree.nodes

for node in shader_nodes:
    if node.label == "Base Color":
        albedo = node
    if node.label == "Map Range":
        roughness = node
    if node.label == "Surface Normal":
        normal = node
    if node.label == "Script":
        script = node
    if node.label == "Multiply":
        operation = node
    if node.label == "Output":
        material_output = node

for input in material_output.inputs:
    if input.name == "Surface":
        material_output_input = input
        break


# Render Normal Map and Alpha Map
## Set up render node tree
if not DEBUG:
    # if nodes.find('Script Generated Render Layers') == -1:
    # Create input render layer node.
    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.name = 'Script Generated Render Layers'

    alpha_file_output = nodes.new(type="CompositorNodeOutputFile")
    alpha_file_output.name = 'alpha_output'
    alpha_link = links.new(render_layers.outputs['Alpha'], alpha_file_output.inputs[0])

    alpha_file_output.base_path = save_path


## Add normal shader link
normal_output_link = shader_links.new(normal.outputs[0], material_output_input)

## Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'frames': []
}

for i in range(0, VIEWS):
    os.makedirs(os.path.join(save_path, f'{i:03d}'), exist_ok=True)

    if RANDOM_VIEWS:
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
            b_empty.rotation_euler = rot
        else:
            rot = np.random.uniform(0, 2 * np.pi, size=3)
            b_empty.rotation_euler = rot
    else:
        # print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
        # scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{int(i * stepsize):03d}')
        b_empty.rotation_euler[2] += math.radians(stepsize)
        rot = b_empty.rotation_euler


    scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}_normal')
    alpha_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + alpha_file_output.name + "_")
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still
    TAG = "0002"
    os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + alpha_file_output.name + "_" + TAG + ".png"),
              os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_alpha" + ".png"))

    frame_data = {
        'file_path': os.path.relpath(scene.render.filepath, start=save_path),
        'rotation': [rot[0], rot[1], rot[2]],
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(save_path + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


## Remove shader link
shader_links.remove(normal_output_link)
## Remove compositor links and nodes
links.remove(alpha_link)
nodes.remove(render_layers)
nodes.remove(alpha_file_output)


# Render light direction map
## Add light direction shader link
direction_output_link = shader_links.new(script.outputs[0], material_output_input)

for i in range(0, VIEWS):
    b_empty.rotation_euler = out_data['frames'][i]['rotation']

    scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}_direction')
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

## Remove shader link
shader_links.remove(direction_output_link)


# Render albedo map
## Add albedo shader link
operation_input0_link = shader_links.new(albedo.outputs[0], operation.inputs[0])
operation_input1_link = shader_links.new(script.outputs[1], operation.inputs[1])
albedo_output_link = shader_links.new(operation.outputs[0], material_output_input)

for i in range(0, VIEWS):
    b_empty.rotation_euler = out_data['frames'][i]['rotation']

    scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}_albedo')
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

## Remove shader link
shader_links.remove(operation_input0_link)
shader_links.remove(operation_input1_link)
shader_links.remove(albedo_output_link)


# Render roughness map
## Add roughness shader link
operation_input0_link = shader_links.new(roughness.outputs[0], operation.inputs[0])
operation_input1_link = shader_links.new(script.outputs[1], operation.inputs[1])
roughness_output_link = shader_links.new(operation.outputs[0], material_output_input)

for i in range(0, VIEWS):
    b_empty.rotation_euler = out_data['frames'][i]['rotation']

    scene.render.filepath = os.path.join(save_path, f'{i:03d}', f'{i:03d}_roughness')
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

## Remove shader link
shader_links.remove(operation_input0_link)
shader_links.remove(operation_input1_link)
shader_links.remove(roughness_output_link)