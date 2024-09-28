"""




    

        

    python blender_render_batch_scans.py -i /path/to/input -o /path/to/output --width 2048 --height 2048
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
import bpy
import argparse


def set_render_settings(    
    resolution_x: int = 4096,
    resolution_y: int = 4096,
    file_format: str = "PNG",  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: str = "16",  # ('8', '16')
    color_mode: str = "RGBA",  # ('RGB', 'RGBA', ...)
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

    render.engine = 'CYCLES'
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
            bpy.context.scene.render.filepath = os.path.join(render_fn)
            if depth_file_output:
                depth_file_output.base_path = os.path.join(depth_fn)
            bpy.ops.render.render(write_still=True)
    subject.rotation_euler = original_rotation


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
    for model in models:
        depth_file_output = set_render_settings(width, height)
        obj = load_textured_mesh(str(model), 0.8)
        rotate360_and_render(
            f"{output_dir}/{model.stem}",
            36,
            obj,
            depth_file_output
        )
        for obj in bpy.context.scene.objects:
            # Check if the object is a mesh
            if obj.type == 'MESH':
                # If it is, delete it
                bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render 3D models.")
    parser.add_argument("-i", "--input_dir", type=str, help="Directory containing input .obj files")
    parser.add_argument("-i", "--output_dir", type=str, help="Directory to save rendered images")
    parser.add_argument("-w", "--width", type=int, default=4096, help="Width of the rendered images")
    parser.add_argument("-h", "--height", type=int, default=4096, help="Height of the rendered images")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    width = args.width
    height = args.height
    render_folder(input_dir, output_dir, width, height)
