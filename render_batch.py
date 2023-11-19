import argparse
import sys
import math
import bpy
from pathlib import Path


def render_model(
    model_file: str,
    output_dir: str,
    resolution: int = 1024,
    num_views_z: int = 12,
    num_views_x: int = 6,
    engine: str = 'BLENDER_EEVEE',
    file_format: str = 'PNG',  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: int = '8',  # ('8', '16')
    color_mode: str = 'RGBA',  # ('RGB', 'RGBA', ...)
    max_dim: int = 0.5
):
    """ Renders the given obj file with rotation. The renderings are saved in a
    subfolder.

    Args:
        model_file (str): Path to obj file
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

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = file_format
    depth_file_output.format.color_depth = color_depth
    if file_format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
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

        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = ''
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = file_format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    albedo_file_output.base_path = ''
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = file_format
    albedo_file_output.format.color_mode = color_mode
    albedo_file_output.format.color_depth = color_depth
    links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

    # Create id map output nodes
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = 'ID Output'
    id_file_output.base_path = ''
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = file_format
    id_file_output.format.color_depth = color_depth

    if file_format == 'OPEN_EXR':
        links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
    else:
        id_file_output.format.color_mode = 'BW'

        divide_node = nodes.new(type='CompositorNodeMath')
        divide_node.operation = 'DIVIDE'
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2**int(color_depth)

        links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_scene.obj(filepath=model_file)

    obj = bpy.context.selected_objects[0]

    context.view_layer.objects.active = obj

    # set object origin to center of mass
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')

    # move object to 000
    bpy.ops.object.location_clear(clear_delta=False)
    bpy.ops.object.location_clear(clear_delta=True)
    bpy.context.object.location = (0, 0, 0)

    # scale object
    max_dim_current = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
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
    light2.energy = 0.015
    bpy.data.objects['Sun'].rotation_euler = \
        bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    stepsize_z = 360.0 / num_views_z
    stepsize_x = 180.0 / num_views_x

    fp = model_file.split('/')[-1].split('.')[0]
    fp = fp.replace(' ', '_').replace('-', '_')
    bpy.ops.view3d.camera_to_view_selected()
    for z in range(0, num_views_z+1):
        angle_z = math.radians(z*stepsize_z)
        for x in range(0, num_views_x+1):
            angle_x = math.radians(x*stepsize_x)
            render_fp = f"{output_dir}/{fp}_z_{int(z*stepsize_z):03d}_x_{int(x*stepsize_x):03d}"

            scene.render.filepath = render_fp
            depth_file_output.file_slots[0].path = render_fp + "_depth"
            normal_file_output.file_slots[0].path = render_fp + "_normal"
            albedo_file_output.file_slots[0].path = render_fp + "_albedo"
            id_file_output.file_slots[0].path = render_fp + "_id"

            bpy.ops.render.render(write_still=True)  # render still

            cam_empty.rotation_euler = (angle_x, 0, angle_z)


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


for model_file in Path(input_dir).rglob('*.obj'):
    render_model(str(model_file), str(output_dir), resolution, 1, 1, engine)
