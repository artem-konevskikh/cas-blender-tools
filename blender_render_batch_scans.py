"""
This script batch renders 3D models from .obj files using Blender. It provides functions to set render settings, load textured meshes, rotate and render objects, and process all models in a specified directory.

Functions:
    set_render_settings(resolution_x: int, resolution_y: int, file_format: str, color_depth: str, color_mode: str) -> bpy.types.CompositorNodeOutputFile:
        Sets the render settings for a Blender scene.

    load_textured_mesh(model_file: str, max_dim: float) -> bpy.types.Object:
        Loads a textured mesh from an OBJ file, centers it, scales it to a maximum dimension, and applies various transformations and settings to prepare it for rendering in Blender.

    rotate360_and_render(output_dir: str, rotation_steps: int, subject: bpy.types.Object, depth_file_output: bpy.types.CompositorNodeOutputFile) -> None:

    render_folder(input_dir: str, output_dir: str, width: int, height: int) -> None:

Usage:
    Run this script from the command line with the following arguments:
        -i, --input_dir: Directory containing input .obj files.
        -o, --output_dir: Directory to save rendered images.
        -w, --width: Width of the rendered images (optional, default is 4096).
        -h, --height: Height of the rendered images (optional, default is 4096).

Example:
    blender --background --python blender_render_batch_scans.py -- -i /path/to/input -o /path/to/output -w 2048 -h 2048
"""


import os
from math import radians
from pathlib import Path
import sys
import bpy
import argparse


def set_render_settings(    
    resolution_x: int = 4096,
    resolution_y: int = 4096,
    file_format: str = "PNG",  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: str = "16",  # ('8', '16')
    color_mode: str = "RGB",  # ('RGB', 'RGBA', ...)
):
    """
    Set the render settings for a Blender scene.
    Parameters:
    resolution_x (int): The horizontal resolution of the render. Default is 4096.
    resolution_y (int): The vertical resolution of the render. Default is 4096.
    file_format (str): The file format for the output image. Options are 'PNG', 'OPEN_EXR', 'JPEG'. Default is 'PNG'.
    color_depth (str): The color depth of the output image. Options are '8', '16'. Default is '16'.
    color_mode (str): The color mode of the output image. Options are 'RGB', 'RGBA', etc. Default is 'RGBA'.
    Returns:
    bpy.types.CompositorNodeOutputFile: The depth file output node.
    """
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = 'BLENDER_EEVEE'
    render.image_settings.color_mode = color_mode
    render.image_settings.color_depth = color_depth
    render.image_settings.file_format = file_format
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = 100
    render.film_transparent = True

    scene.use_nodes = True
    context.window.view_layer.use_pass_normal = True
    context.window.view_layer.use_pass_diffuse_color = True
    context.window.view_layer.use_pass_z = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = ""
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = file_format
    depth_file_output.format.color_depth = color_depth
    if file_format == "OPEN_EXR":
        links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_mode = "BW"
        # Remap as other types can not represent the full range of depth.
        map = nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily,
        # try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [1.4]
        map.use_min = True
        map.min = [0]
        map.use_max = True
        map.max = [255]

        links.new(render_layers.outputs["Depth"], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

    return depth_file_output


def load_textured_mesh(model_file: str, max_dim: float = 0.5):
    """
    Loads a textured mesh from an OBJ file, centers it, scales it to a maximum dimension, and applies various 
    transformations and settings to prepare it for rendering in Blender.
    Args:
        model_file (str): The file path to the OBJ model to be imported.
        max_dim (float, optional): The maximum dimension to scale the object to. Defaults to 0.5.
    Returns:
        bpy.types.Object: The imported and processed Blender object.
    """

    # Import textured mesh
    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.import_scene.obj(filepath=model_file)

    obj = bpy.context.selected_objects[0]

    bpy.context.view_layer.objects.active = obj

    # set object origin to center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="BOUNDS")

    # move object to 000
    bpy.ops.object.location_clear(clear_delta=False)
    bpy.ops.object.location_clear(clear_delta=True)
    bpy.context.object.location = (0, 0, 0)

    # scale object
    max_dim_current = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    ratio = max_dim / max_dim_current
    bpy.ops.transform.resize(value=(ratio, ratio, ratio))
    bpy.ops.object.transform_apply(scale=True)

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes["Principled BSDF"]
        node.inputs["Specular"].default_value = 0.001

    # Remove double vertices to improve mesh quality
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Adds edge split filter
    bpy.ops.object.modifier_add(type="EDGE_SPLIT")
    bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Set objekt IDs
    obj.pass_index = 1
    return obj


def set_camera():
    # Place camera
    cam = bpy.context.scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.rotation_euler = (radians(90), 0, radians(180))
    cam.constraints.new(type='TRACK_TO')
    cam.constraints['Track To'].target = bpy.data.objects.new("Empty", None)
    cam.constraints['Track To'].target.location = (0, 0, 0)
    bpy.context.collection.objects.link(cam.constraints['Track To'].target)
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def add_lights():
    # Create lights
    light_data_1 = bpy.data.lights.new(name="Light1", type='AREA')
    light_data_1.shape = 'DISK'
    light_data_1.size = 1
    light_data_1.energy = 150
    light_data_1.color = (1, 1, 1)
    light_data_1.use_shadow = True
    light_object_1 = bpy.data.objects.new(name="Light1", object_data=light_data_1)
    light_object_1.location = (0, 0, 2.14)
    light_object_1.scale = (0.6, 0.6, 0.6)
    bpy.context.collection.objects.link(light_object_1)
    
    light_data_2 = bpy.data.lights.new(name="Light2", type='AREA')
    light_data_2.shape = 'DISK'
    light_data_2.size = 1
    light_data_2.energy = 10
    light_data_2.color = (1, 1, 1)
    light_data_2.use_shadow = True
    light_object_2 = bpy.data.objects.new(name="Light2", object_data=light_data_2)
    light_object_2.location = (0, 0, 1.45)
    light_object_2.scale = (2.7, 2.7, 2.7)
    bpy.context.collection.objects.link(light_object_2)
    
    light_data_3 = bpy.data.lights.new(name="Light3", type='AREA')
    light_data_3.shape = 'DISK'
    light_data_3.size = 6
    light_data_3.energy = 10
    light_data_3.color = (1, 1, 1)
    light_data_3.use_shadow = True
    light_object_3 = bpy.data.objects.new(name="Light3", object_data=light_data_3)
    light_object_3.location = (0, 0, -1.13)
    bpy.context.collection.objects.link(light_object_3)


def rotate360_and_render(    
    output_dir: str,
    rotation_steps: int = 36,
    subject = bpy.context.object,
    depth_file_output = None,
):
    """
    Rotates the given subject 360 degrees in both X and Z axes and renders images at each step.
    Args:
        output_dir (str): The directory where the rendered images and depth files will be saved.
        rotation_steps (int, optional): The number of steps to divide the 360-degree rotation into. Defaults to 36.
        subject (bpy.types.Object, optional): The Blender object to be rotated and rendered. Defaults to the currently active object.
        depth_file_output (bpy.types.CompositorNodeOutputFile, optional): The Blender node for outputting depth files. Defaults to None.
    Returns:
        None
    """
    original_rotation = subject.rotation_euler
    
    for step_x in range(0, rotation_steps):
        for step_z in range(0, rotation_steps):
            subject.rotation_euler[0] = radians(step_x * (360.0 / rotation_steps))
            subject.rotation_euler[2] = radians(step_z * (360.0 / rotation_steps))
            render_fn = os.path.join(output_dir, f"render_x{step_x}_z{step_z}.jpg")
            depth_fn = os.path.join(output_dir, f"depth_x{step_x}_z{step_z}")
            # render_fn = f"{output_dir}_render_x{step_x}_z{step_z}.png"
            # depth_fn = f"{output_dir}_depth_x{step_x}_z{step_z}.png"
            bpy.context.scene.render.filepath = os.path.join(render_fn)
            if depth_file_output:
                depth_file_output.base_path = os.path.join(depth_fn)
            bpy.ops.render.render(write_still=True)
    subject.rotation_euler = original_rotation


def delete_all_meshes():
    for obj in bpy.context.scene.objects:
            # Check if the object is a mesh
            if obj.type == 'MESH':
                # If it is, delete it
                bpy.data.objects.remove(obj, do_unlink=True)


def render_folder(
    input_dir: str,
    output_dir: str,
    width: int,
    height: int,
):
    """
    Renders all .obj models in the specified input directory and saves the rendered images to the output directory.
    Args:
        input_dir (str): The directory containing .obj models to be rendered.
        output_dir (str): The directory where rendered images will be saved.
        width (int): The width for the rendered images.
        height (int): The height for the rendered images.
    Returns:
        None
    """
    models = Path(input_dir).rglob("*.obj")
    depth_file_output = set_render_settings(width, height)
    add_lights()
    set_camera()
    delete_all_meshes()
    for model in models:
        if os.path.exists(f"{output_dir}/{model.stem}"):
            print(f"Skipping {model.stem}")
            continue
        
        print(f"Rendering {model.stem}")
        obj = load_textured_mesh(str(model), 0.7)
        
        rotate360_and_render(
            f"{output_dir}/{model.stem}",
            18,
            obj,
            depth_file_output
        )
        delete_all_meshes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render 3D models.")
    parser.add_argument("-i", "--input_dir", type=str, help="Directory containing input .obj files")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to save rendered images")
    parser.add_argument("--width", type=int, default=4096, help="Width of the rendered images")
    parser.add_argument("--height", type=int, default=4096, help="Height of the rendered images")

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir
    width = args.width
    height = args.height
    render_folder(input_dir, output_dir, width, height)
