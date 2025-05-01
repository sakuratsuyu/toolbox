# Blender Toolbox

- `parse_textures.py`
    - Parse and print the node tree of a material of an object.
    - Usage
        1. load it in Blender.
        2. change `obj = bpy.data.objects['Brick_flat_02*08.002']` to select the object.
        3. run it.
- lego
    - `lego/renderer.py`
        - Used with `lego/postprocess.py`.
        - Render a model at the world origin, with a surrounded camera and a light.
        - Render `render`, `alpha`, `depth`, `normal`, `position`, `albedo` and `roughness`.
        - Usage
            1. load it in Blender and run it.
            2. change the settings in `lego/renderer.py`.
            3. run it.
    - `lego/postprocess.py`
        - Used with `lego/renderer.py`.
        - Postprocess the rendered images.
        - Usage
            - `python postprocess.py --image_path <image_directory>`
    - `lego/renderer_index.py`
        - Used with `lego/postprocess_index.py`.
        - Render a model at the world origin, with a surrounded camera and a light.
        - Render `render`, `alpha`, `depth`, `normal`, `position` and `material_index`.
        - Usage
            1. load it in Blender and run it.
            2. change the settings in `lego/renderer_index.py`.
            3. run it.
    - `lego/postprocess_index.py`
        - Used with `lego/renderer_index.py`.
        - Postprocess the rendered images.
        - Change material settings.
        - Usage
            - `python postprocess_index.py --image_path <image_directory>`