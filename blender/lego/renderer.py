import os
import json
import bpy
import numpy as np
import mathutils
import math

DEBUG = False

# Settings
RESOLUTION = 800
NUM_VIEWS = 4
NUM_LIGHT_POSITIONS = 4
RANDOM_VIEWS = False
UPPER_VIEWS = True
ROTATION_LIGHT = False
RESULTS_PATH = f"lego_{NUM_VIEWS:03d}_L{NUM_LIGHT_POSITIONS:03d}"
COLOR_DEPTH = 8
FORMAT = "PNG"

CAMERA_POISTION_RADIUS = 4.0
LIGHT_POSITION_DISTANCE = 5.0
LIGHT_POSITION_RANGE_X = 1
LIGHT_POSITION_RANGE_Y = 1
LIGHT_POSITION_RANGE_Z = 1
LIGHT_INTENISTY = 10
POINTS = np.array([[-1.0,  1.0, 0.0], [-1.0, -1.0, 0.0],
                   [ 1.0, -1.0, 0.0], [ 1.0,  1.0, 0.0]])


scene = bpy.context.scene
materials = bpy.data.materials
save_path = bpy.path.abspath(f"//{RESULTS_PATH}")
os.makedirs(save_path, exist_ok=True)


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def initilize():
    scene.render.use_persistent_data = True # render optimizations
    scene.frame_start = 0
    scene.frame_end = NUM_VIEWS * NUM_LIGHT_POSITIONS - 1
    scene.render.filepath = os.path.join(save_path, "image_")
    scene.render.image_settings.file_format = str(FORMAT)
    scene.render.image_settings.color_depth = str(COLOR_DEPTH)
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True

def add_file_output(render_layers, nodes, links):
    alpha_file_output = nodes.new(type="CompositorNodeOutputFile")
    alpha_file_output.name = "alpha_output"
    alpha_link = links.new(render_layers.outputs["Alpha"], alpha_file_output.inputs[0])
    alpha_file_output.base_path = save_path
    alpha_file_output.format.file_format = "OPEN_EXR"
    alpha_file_output.file_slots[0].path = "alpha_"

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.name = "depth_output"
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_link = links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
    depth_file_output.base_path = save_path
    depth_file_output.file_slots[0].path = "depth_"

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.name = "normal_output"
    normal_file_output.format.file_format = "OPEN_EXR"
    normal_link = links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])
    normal_file_output.base_path = save_path
    normal_file_output.file_slots[0].path = "normal_"

    position_file_output = nodes.new(type="CompositorNodeOutputFile")
    position_file_output.name = "position_output"
    position_file_output.format.file_format = "OPEN_EXR"
    position_link = links.new(render_layers.outputs["Position"], position_file_output.inputs[0])
    position_file_output.base_path = save_path
    position_file_output.file_slots[0].path = "position_"

    render_file_output = nodes.new(type="CompositorNodeOutputFile")
    render_file_output.name = "render_output"
    render_file_output.format.file_format = "OPEN_EXR"
    render_link = links.new(render_layers.outputs["Image"], render_file_output.inputs[0])
    render_file_output.base_path = save_path
    render_file_output.file_slots[0].path = "render_"

    gbuffer_links = [alpha_link, depth_link, normal_link, position_link]
    gbuffer_outputs = [alpha_file_output, depth_file_output, normal_file_output, position_file_output]
    return render_file_output, render_link, gbuffer_outputs, gbuffer_links

def set_camera():
    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scene.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        # scn.objects.active = b_empty
        return b_empty

    camera = scene.objects["Camera"]
    camera.location = (0, 4.0, 0.0)
    if camera.constraints.find("Track To") == -1:
        cam_constraint = camera.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        b_empty = parent_obj_to_camera(camera)
        cam_constraint.target = b_empty
    else:
        cam_constraint = camera.constraints["Track To"]
        b_empty = cam_constraint.target

    STEPSIZE_VIEW = 1.0 / NUM_VIEWS
    if RANDOM_VIEWS:
        if UPPER_VIEWS:
            theta = np.random.uniform(0, np.pi / 2, NUM_VIEWS)
            phi = np.random.uniform(0, 2 * np.pi, NUM_VIEWS + 1)[:-1]
        else:
            theta = np.random.uniform(0, np.pi, NUM_VIEWS + 1)[:-1]
            phi = np.random.uniform(0, 2 * np.pi, NUM_VIEWS + 1)[:-1]
    else:
        sqrt = math.sqrt(NUM_VIEWS)
        assert sqrt == int(sqrt)
        sqrt = int(sqrt)
        theta = np.linspace(0, np.pi / 2, sqrt + 1)[:-1] + 1e-3
        phi = np.linspace(0, 2 * np.pi, sqrt + 1)[:-1]
        # n = 2
        # m = 4
        # theta = np.linspace(0, np.pi / 2, n + 1)[:-1] + 1e-3
        # phi = np.linspace(0, 2 * np.pi, m + 1)[:-1]
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.flatten()
        phi = phi.flatten()

    position = CAMERA_POISTION_RADIUS * np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    
    return camera, position

def set_lights():
    light = scene.objects["Light"]
    light_material = light.data.materials[0]
    light_shader_nodes = light_material.node_tree.nodes
    emission = None
    for node in light_shader_nodes:
        if node.label == "emission":
            emission = node
    light_intensity = emission.inputs[1].default_value

    STEPSIZE_LIGHT = 1.0 / NUM_LIGHT_POSITIONS
    if ROTATION_LIGHT:
        sqrt = math.sqrt(NUM_LIGHT_POSITIONS)
        assert sqrt == int(sqrt)
        sqrt = int(sqrt)
        light_theta = np.linspace(0, np.pi / 2, sqrt)
        light_phi = np.linspace(0, 2 * np.pi, sqrt + 1)[:-1]
        light_theta, light_phi = np.meshgrid(light_theta, light_phi)
        light_theta = light_theta.flatten()
        light_phi = light_phi.flatten()

        direction = np.stack([np.sin(light_theta) * np.cos(light_phi), np.sin(light_theta) * np.sin(light_phi), np.cos(light_theta)], axis=-1)
        z = np.array([0, 0, 1])[None, :]
        tangent = z - direction * np.sum(z * direction, axis=-1, keepdims=True)
        tangent = tangent / np.linalg.norm(tangent, keepdims=True)
        bitangent = np.cross(direction, tangent)
        light_rotation = np.stack([
            tangent,
            bitangent,
            direction
        ], axis=-1)
        light_translation = np.array([0, 0, LIGHT_POSITION_DISTANCE])
    else:
        light_theta = np.zeros(NUM_LIGHT_POSITIONS)
        light_phi = np.zeros(NUM_LIGHT_POSITIONS)
        if NUM_LIGHT_POSITIONS == 1:
            x = np.zeros(NUM_LIGHT_POSITIONS)
            y = np.zeros(NUM_LIGHT_POSITIONS)
            z = np.ones(NUM_LIGHT_POSITIONS) * LIGHT_POSITION_DISTANCE
        else:
            x = np.random.uniform(-1, 1, NUM_LIGHT_POSITIONS) * LIGHT_POSITION_RANGE_X
            y = np.random.uniform(-1, 1, NUM_LIGHT_POSITIONS) * LIGHT_POSITION_RANGE_Y
            z = np.random.uniform(-1, 1, NUM_LIGHT_POSITIONS) * LIGHT_POSITION_RANGE_Z + LIGHT_POSITION_DISTANCE
        light_rotation = np.eye(3)[None, :, :].repeat(NUM_LIGHT_POSITIONS, axis=0)
        light_translation = np.stack([x, y, z], axis=-1)


    light_matrix = np.concatenate([light_rotation, light_translation[:, :, None]], axis=-1)
    line = np.array([0, 0, 0, 1])[None, None, :].repeat(NUM_LIGHT_POSITIONS, axis=0)
    light_matrix = np.concatenate([light_matrix, line], axis=1)
    light_vertices_position = np.einsum('nij,kj->nki', light_rotation, POINTS) + light_translation[:, None, :]

    return light, light_intensity, light_matrix, light_vertices_position

def main():
    initilize()

    # Create input render layer node.
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    nodes = tree.nodes
    render_layers = nodes.new(type="CompositorNodeRLayers")
    render_layers.name = "Script Generated Render Layers"
    render_file_output, render_link, gbuffer_outputs, gbuffer_links = add_file_output(render_layers, nodes, links)


    camera, camera_position = set_camera()
    light, light_intensity, light_matrix, light_vertices_position = set_lights()


    # Parameter
    out_data = {
        "camera_angle_x": camera.data.angle_x,
        "frames": []
    }

    for i in range(0, NUM_VIEWS):
        # camera.location = [camera_position[i, 0], camera_position[i, 1], camera_position[i, 2]]
        camera.location = [camera_position[i, 1], -camera_position[i, 0], camera_position[i, 2]]

        for j in range(0, NUM_LIGHT_POSITIONS):
            # os.makedirs(os.path.join(save_path, f"{i:03d}"), exist_ok=True)
            cnt = i * NUM_LIGHT_POSITIONS + j

            light.matrix_world = mathutils.Matrix(light_matrix[j])
            
            camera.keyframe_insert(data_path="location", index=-1, frame=cnt)
            light.keyframe_insert(data_path="location", index=-1, frame=cnt)
            light.keyframe_insert(data_path="rotation_euler", index=-1, frame=cnt)
            bpy.context.scene.frame_set(cnt)

            frame_data = {
                "file_path": os.path.join(f"{i:03d}_L{j:03d}", f"{i:03d}_L{j:03d}"),
                "light_intensity": light_intensity,
                "light_vertices_position": light_vertices_position[j].tolist(),
                "transform_matrix": listify_matrix(camera.matrix_world),
            }
            out_data["frames"].append(frame_data)

    if not DEBUG:
        with open(save_path + "/" + "transforms.json", "w") as out_file:
            json.dump(out_data, out_file, indent=4)


    # Render
    # buffer_names = ["brdf", "albedo", "roughness"]
    buffer_names = ["brdf"]
    # render_names = ["render", "albedo", "roughness"]
    render_names = ["render"]

    gbuffer_flag = False

    for (buffer_name, render_name) in zip(buffer_names, render_names):
        material_links = []

        for material in materials:
            shader_tree = material.node_tree
            shader_links = shader_tree.links
            shader_nodes = shader_tree.nodes

            albedo = None
            buffer = None
            material_output = None

            for node in shader_nodes:
                if node.label == "albedo":
                    albedo = node
                if node.label == buffer_name:
                    buffer = node
                if node.label == "output":
                    material_output = node

            if albedo is None:
                continue

            for input in material_output.inputs:
                if input.name == "Surface":
                    material_output_input = input
                    break

            ## Add buffer shader link
            material_links.append(shader_links.new(buffer.outputs[0], material_output_input))

        render_file_output.file_slots[0].path = render_name + "_"
        render_file_output.format.file_format = "OPEN_EXR"

        if not DEBUG:
            bpy.ops.render.render(animation=True)

        for material in materials:
            shader_tree = material.node_tree
            shader_links = shader_tree.links
            shader_nodes = shader_tree.nodes

            albedo = None
            material_output = None

            for node in shader_nodes:
                if node.label == "albedo":
                    albedo = node
            
            if albedo is None:
                continue

            ## Remove shader link
            shader_links.remove(material_links.pop(0))

        # Remove compositor links and nodes
        if gbuffer_flag == False:
            gbuffer_flag = True
            for link in gbuffer_links:
                links.remove(link)
            for node in gbuffer_outputs:
                nodes.remove(node)

    # Remove compositor links and nodes
    links.remove(render_link)
    nodes.remove(render_file_output)
    nodes.remove(render_layers)

    if not DEBUG:
        for i in range(0, NUM_VIEWS * NUM_LIGHT_POSITIONS):
            os.remove(os.path.join(save_path, f"image_{i:04d}.png"))

if __name__ == "__main__":
    main()