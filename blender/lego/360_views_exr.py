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

TAG = "0000"
VIEWS = 100
RESOLUTION = 800
RESULTS_PATH = f'images_{VIEWS:03d}'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
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


scene.frame_start = 0
scene.frame_end = VIEWS - 1
scene.render.filepath = os.path.join(save_path, 'image_')

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
cam.location = (0, 5.0, 0.0)
if cam.constraints.find("Track To") == -1:
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
else:
    cam_constraint = cam.constraints["Track To"]
    b_empty = cam_constraint.target



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
    alpha_file_output.file_slots[0].path = 'alpha_'

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.name = 'normal_output'
    normal_file_output.format.file_format = 'OPEN_EXR'
    normal_link = links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    normal_file_output.base_path = save_path
    normal_file_output.file_slots[0].path = 'normal_'

    position_file_output = nodes.new(type="CompositorNodeOutputFile")
    position_file_output.name = 'position_output'
    position_file_output.format.file_format = 'OPEN_EXR'
    position_link = links.new(render_layers.outputs['Position'], position_file_output.inputs[0])
    position_file_output.base_path = save_path
    position_file_output.file_slots[0].path = 'position_'

    image_file_output = nodes.new(type="CompositorNodeOutputFile")
    image_file_output.name = 'image_output'
    image_file_output.format.file_format = 'OPEN_EXR'
    image_link = links.new(render_layers.outputs['Image'], image_file_output.inputs[0])
    image_file_output.base_path = save_path
    image_file_output.file_slots[0].path = 'image_'


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
        else:
            rot = np.random.uniform(0, 2 * np.pi, size=3)
    else:
        rot = [math.radians(30
        ), 0, math.radians(stepsize * i)]

    b_empty.rotation_euler = rot
    b_empty.keyframe_insert(data_path="rotation_euler", index=-1, frame=i)
    bpy.context.scene.frame_set(i)

    frame_data = {
        'file_path': os.path.join(f'{i:03d}', f'{i:03d}'),
        'transform_matrix': listify_matrix(cam.matrix_world),
    }
    out_data['frames'].append(frame_data)


if not DEBUG:
    with open(save_path + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)



materials = bpy.data.materials
material_links = []



# Render
for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    brdf = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "brdf":
            brdf = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    for input in material_output.inputs:
        if input.name == "Surface":
            material_output_input = input
            break

    ## Add brdf shader link
    material_links.append(shader_links.new(brdf.outputs[0], material_output_input))

image_file_output.file_slots[0].path = 'render_'
bpy.ops.render.render(animation=True)

# for i in range(0, VIEWS):
#     b_empty.rotation_euler = out_data['frames'][i]['rotation']
#     scene.render.filepath = out_data['frames'][i]['file_path']

#     alpha_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + alpha_file_output.name + "_")
#     normal_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + normal_file_output.name + "_")

#     if DEBUG:
#         break
#     else:
#         bpy.ops.render.render(write_still=True)  # render still

#     os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + alpha_file_output.name + "_" + TAG + ".png"),
#             os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_alpha" + ".png"))
#     os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + normal_file_output.name + "_" + TAG + ".exr"),
#             os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_normal" + ".exr"))

for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    brdf = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "brdf":
            brdf = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    ## Remove shader link
    shader_links.remove(material_links.pop(0))

## Remove compositor links and nodes
links.remove(alpha_link)
links.remove(normal_link)
links.remove(position_link)
nodes.remove(alpha_file_output)
nodes.remove(normal_file_output)
nodes.remove(position_file_output)


# Render albedo map
for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    roughness = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "roughness":
            roughness = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    for input in material_output.inputs:
        if input.name == "Surface":
            material_output_input = input
            break

    ## Add albedo shader link
    material_links.append(shader_links.new(albedo.outputs[0], material_output_input))

image_file_output.file_slots[0].path = 'albedo_'
bpy.ops.render.render(animation=True)

# for i in range(0, VIEWS):
#     b_empty.rotation_euler = out_data['frames'][i]['rotation']
#     scene.render.filepath = out_data['frames'][i]['file_path']

#     image_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + "albedo" + "_")
#     if DEBUG:
#         break
#     else:
#         bpy.ops.render.render(write_still=True)  # render still
#     os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + "albedo" + "_" + TAG + ".exr"),
#             os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_albedo" + ".exr"))

for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    roughness = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "roughness":
            roughness = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    ## Remove shader link
    shader_links.remove(material_links.pop(0))



# Render roughness map
for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    roughness = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "roughness":
            roughness = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    for input in material_output.inputs:
        if input.name == "Surface":
            material_output_input = input
            break

    ## Add roughness shader link
    material_links.append(shader_links.new(roughness.outputs[0], material_output_input))

image_file_output.file_slots[0].path = 'roughness_'
bpy.ops.render.render(animation=True)

# for i in range(0, VIEWS):
#     b_empty.rotation_euler = out_data['frames'][i]['rotation']
#     scene.render.filepath = out_data['frames'][i]['file_path']

#     image_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + "roughness" + "_")
#     if DEBUG:
#         break
#     else:
#         bpy.ops.render.render(write_still=True)  # render still
#     os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + "roughness" + "_" + TAG + ".exr"),
#               os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_roughness" + ".exr"))

for material in materials:
    shader_tree = material.node_tree
    shader_links = shader_tree.links
    shader_nodes = shader_tree.nodes

    albedo = None
    roughness = None
    material_output = None

    for node in shader_nodes:
        if node.label == "albedo":
            albedo = node
        if node.label == "roughness":
            roughness = node
        if node.label == "output":
            material_output = node
    
    if albedo is None:
        continue

    ## Remove shader link
    shader_links.remove(material_links.pop(0))



# Render viewdir map
# for material in materials:
#     shader_tree = material.node_tree
#     shader_links = shader_tree.links
#     shader_nodes = shader_tree.nodes

#     albedo = None
#     viewdir = None
#     material_output = None

#     for node in shader_nodes:
#         if node.label == "albedo":
#             albedo = node
#         if node.label == "viewdir":
#             viewdir = node
#         if node.label == "output":
#             material_output = node
    
#     if albedo is None:
#         continue

#     for input in material_output.inputs:
#         if input.name == "Surface":
#             material_output_input = input
#             break

#     ## Add viewdir link
#     material_links.append(shader_links.new(viewdir.outputs[1], material_output_input))

# image_file_output.file_slots[0].path = 'viewdir_'
# bpy.ops.render.render(animation=True)

# # for i in range(0, VIEWS):
# #     b_empty.rotation_euler = out_data['frames'][i]['rotation']
# #     scene.render.filepath = out_data['frames'][i]['file_path']

# #     image_file_output.file_slots[0].path = os.path.join(f'{i:03d}', f'{i:03d}' + "_" + "viewdir" + "_")
# #     if DEBUG:
# #         break
# #     else:
# #         bpy.ops.render.render(write_still=True)  # render still
# #     os.rename(os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_" + "viewdir" + "_" + TAG + ".exr"),
# #             os.path.join(save_path, f'{i:03d}', f'{i:03d}' + "_viewdir" + ".exr"))

# for material in materials:
#     shader_tree = material.node_tree
#     shader_links = shader_tree.links
#     shader_nodes = shader_tree.nodes

#     albedo = None
#     viewdir = None
#     material_output = None

#     for node in shader_nodes:
#         if node.label == "albedo":
#             albedo = node
#         if node.label == "viewdir":
#             viewdir = node
#         if node.label == "output":
#             material_output = node
    
#     if albedo is None:
#         continue

#     ## Remove shader link
#     shader_links.remove(material_links.pop(0))



## Remove image compositor links and nodes
links.remove(image_link)
nodes.remove(image_file_output)
nodes.remove(render_layers)

for i in range(0, VIEWS):
    os.remove(os.path.join(save_path, f'image_{i:04d}.png'))