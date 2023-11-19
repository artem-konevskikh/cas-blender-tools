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
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Albedo.jpg'
    tex_image.image = bpy.data.images.load(tex_path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'],
                            tex_image.outputs['Color'])

    disp_path = '../stone_floor_tkkkeicew/tkkkeicew_8K_Displacement.exr'
    tex_disp = bpy.data.textures.new('Disp', type='IMAGE')
    tex_disp.image = bpy.data.images.load(disp_path)

    for model_file in Path(input_dir).rglob('*.obj'):
        # Import textured mesh
        bpy.ops.import_scene.obj(filepath=str(model_file))

        obj = bpy.context.selected_objects[0]

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # disp_mod = obj.modifiers.new("Displace", type='DISPLACE')
        # disp_mod.strength = 0.2
        # # disp_mod.mid_level = 0
        # disp_mod.texture = tex_disp

        context.view_layer.objects.active = obj

        # set object origin to center of mass
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME',
                                  center='BOUNDS')

        # move object to 000
        bpy.ops.object.location_clear(clear_delta=False)
        bpy.ops.object.location_clear(clear_delta=True)
        bpy.context.object.location = (0, 0, 0)

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
        toggle_shading(context.active_object)

        i = int(str(model_file).split('_')[-1].split('.')[0])
        cam_empty.rotation_euler[2] = math.radians(i)

        render_fp = f"{output_dir}/frame_{i:05d}"
        scene.render.filepath = render_fp
        bpy.ops.render.render(write_still=True)

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
