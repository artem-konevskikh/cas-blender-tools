import argparse
import sys
import math
import bpy
from pathlib import Path


def set_shading(object, OnOff=True):
    """ Set the shading mode of an object
        True means turn smooth shading on.
        False means turn smooth shading off.
    """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    polygons.foreach_set('use_smooth',  [OnOff] * len(polygons))
    object.data.update()


def toggle_shading(object):
    """ Toggle the shading mode of an object """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    for polygon in polygons:
        polygon.use_smooth = not polygon.use_smooth
    object.data.update()


def render_models(
    input_dir: str,
    output_dir: str,
    resolution: int = 1024,
    engine: str = 'BLENDER_EEVEE',
    file_format: str = 'PNG',  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: int = '8',  # ('8', '16')
    color_mode: str = 'RGBA',  # ('RGB', 'RGBA', ...)
    max_dim: int = 0.5
):
    """ Renders the given obj file with rotation. The renderings are saved in a
    subfolder.

    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        resolution (int, optional): Resolution of rendered image. Defaults to 1024.
        num_views_z (int, optional): Number of rotations around z-axis. Defaults to 12.
        num_views_x (int, optional): Number of rotations around x-axis. Defaults to 6.
        engine (str, optional): Blender internal render engine. Possible values 'BLENDER_EEVEE', 'CYCLES'. Defaults to 'BLENDER_EEVEE'.
        file_format (str, optional): Rendered image format. Possible values 'PNG', 'OPEN_EXR', 'JPEG'.Defaults to 'PNG'.
        color_depth (int, optional): Color depth of rendered image. Possible values '8', '16'. Defaults to '8'.
        color_mode (str, optional): Color mode of rendered image. Possible values 'RGB', 'RGBA'. Defaults to 'RGBA'.
        max_dim (int, optional): Maximum dimension of the object. Needed to fit object in frame. Defaults to 0.5.
    """
    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = engine
    render.image_settings.color_mode = color_mode
    render.image_settings.color_depth = color_depth
    render.image_settings.file_format = file_format
    render.resolution_x = resolution
    render.resolution_y = resolution
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
    render_layers = nodes.new('CompositorNodeRLayers')

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 10.0

    # Add another light source so stuff facing away from light
    # is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.15
    bpy.data.objects['Sun'].rotation_euler = \
        bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects['Camera']
    cam.location = (-1.6, 0, 0)
    # cam.data.lens = 35
    # cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    bpy.ops.view3d.camera_to_view_selected()

    mat = bpy.data.materials.new(name="New_Mat")
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links

    # Reuse the material output node that is created by default
    material_output = mat_nodes.get("Material Output")
    # Create the Principled BSDF node
    bsdf = mat_nodes["Principled BSDF"]
    mat_links.new(material_output.inputs["Surface"],
                  bsdf.outputs["BSDF"])
    # Create the Texture Coordinate node
    mapping_node = mat_nodes.new("ShaderNodeMapping")
    tex_coordinate = mat_nodes.new("ShaderNodeTexCoord")
    mat_links.new(mapping_node.inputs["Vector"],
                  tex_coordinate.outputs["UV"])

    # Create the Image Texture node
    tex_image = mat_nodes.new('ShaderNodeTexImage')
    tex_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Albedo.jpg'
    tex_image.image = bpy.data.images.load(tex_path)
    mat_links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    mat_links.new(tex_image.inputs["Vector"],
                  mapping_node.outputs["Vector"])

    # Create Image Texture node and load the displacement texture.
    # You need to add the actual path to the texture.
    disp_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Displacement.exr'
    displacement_tex = mat_nodes.new("ShaderNodeTexImage")
    displacement_tex.image = bpy.data.images.load(disp_path)
    displacement_tex.image.colorspace_settings.name = "Non-Color"

    # Create the Displacement node
    displacement = mat_nodes.new("ShaderNodeDisplacement")
    # displacement.inputs["Scale"].default_value = 0.01

    # Connect the Texture Coordinate node to the displacement texture.
    # This uses the active UV map of the object.
    mat_links.new(displacement_tex.inputs["Vector"],
                  mapping_node.outputs["Vector"])

    # Connect the displacement texture to the Displacement node
    mat_links.new(displacement.inputs["Height"],
                  displacement_tex.outputs["Color"])

    # Connect the Displacement node to the Material Output node
    mat_links.new(material_output.inputs["Displacement"],
                  displacement.outputs["Displacement"])

    # Create the Normal texture node
    norm_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Normal.jpg'
    normal_tex = mat_nodes.new("ShaderNodeTexImage")
    normal_tex.image = bpy.data.images.load(norm_path)
    normal_tex.image.colorspace_settings.name = "Non-Color"
    normal_map = mat_nodes.new("ShaderNodeNormalMap")
    # normal_map.inputs['Strength'].default_value = 0.5
    mat_links.new(normal_map.inputs['Color'], normal_tex.outputs['Color'])
    mat_links.new(bsdf.inputs['Normal'], normal_map.outputs['Normal'])
    mat_links.new(normal_tex.inputs["Vector"],
                  mapping_node.outputs["Vector"])

    # Create the Normal texture node
    rough_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Roughness.jpg'
    roughness_tex = mat_nodes.new("ShaderNodeTexImage")
    roughness_tex.image = bpy.data.images.load(rough_path)
    roughness_tex.image.colorspace_settings.name = "Non-Color"
    mat_links.new(bsdf.inputs['Roughness'], roughness_tex.outputs['Color'])
    mat_links.new(roughness_tex.inputs["Vector"],
                  mapping_node.outputs["Vector"])

    for model_file in Path(input_dir).rglob('*.obj'):
        # Import textured mesh
        bpy.ops.import_scene.obj(filepath=str(model_file))

        obj = bpy.context.selected_objects[0]

        obj.rotation_euler[0] = math.radians(3)
        obj.rotation_euler[1] = math.radians(0.2)

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        context.view_layer.objects.active = obj

        bpy.ops.object.subdivision_set(level=3)

        # set object origin to center of mass
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',
                                  center='BOUNDS')

        # move object to 000
        bpy.ops.object.location_clear(clear_delta=False)
        bpy.ops.object.location_clear(clear_delta=True)
        bpy.context.object.location = (0, 0, 0)

        # obj.select_set(True)
        lm = obj.data.uv_layers.get("LightMap")
        if not lm:
            lm = obj.data.uv_layers.new(name="LightMap")
        lm.active = True
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')  # for all faces
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
        bpy.ops.object.editmode_toggle()
        # obj.select_set(False)

        # scale object
        max_dim_current = max(obj.dimensions.x, obj.dimensions.y,
                              obj.dimensions.z)
        ratio = max_dim/max_dim_current
        bpy.ops.transform.resize(value=(ratio, ratio, ratio))
        bpy.ops.object.transform_apply(scale=True)

        # Possibly disable specular shading
        for slot in obj.material_slots:
            node = slot.material.node_tree.nodes['Principled BSDF']
            node.inputs['Specular'].default_value = 0.05

        # Remove double vertices to improve mesh quality
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Adds edge split filter
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

        # Set objekt IDs
        obj.pass_index = 1
        # toggle_shading(context.active_object)

        i = int(str(model_file).split('_')[-1].split('.')[0])
        cam_empty.rotation_euler[2] = math.radians(i)

        render_fp = f"{output_dir}/frame_{i:05d}"
        scene.render.filepath = render_fp
        bpy.ops.render.render(write_still=True)
        # break
        context.active_object.select_set(True)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')


# parse arguments
parser = argparse.ArgumentParser(
    description='Renders given obj file by rotation a camera around it.')
parser.add_argument('-i', '--input_dir', type=str, default='models',
                    help='The path to load models.')
parser.add_argument('-o', '--output_dir', type=str, default='output',
                    help='The path the output will be dumped to.')
parser.add_argument('-r', '--resolution', type=int, default=1024,
                    help='Resolution of the images.')
parser.add_argument('-e', '--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g.\
                        CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)
input_dir = args.input_dir
output_dir = args.output_dir
resolution = args.resolution
engine = args.engine

# create output dirs for renders, depth maps and normal maps
# Path(output_dir).mkdir(parents=True, exist_ok=True)


render_models(input_dir, output_dir, resolution, engine)
