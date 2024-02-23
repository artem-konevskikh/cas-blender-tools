import glob
import math
from mathutils import Quaternion
import open3d as o3d
import numpy as np
import bpy


def vector_angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_floor_plane(pcd, dist_threshold=0.02):
    plane_model, _ = pcd.segment_plane(
        distance_threshold=dist_threshold, ransac_n=3, num_iterations=1000
    )
    return plane_model


def get_mesh_rotation(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    floor = get_floor_plane(pcd)
    a, b, c, d = floor

    # Translate plane to coordinate center
    pcd.translate((0, -d / c, 0))

    # Calculate rotation angle between plane normal & z-axis
    plane_normal = tuple(floor[:3])
    z_axis = (0, 0, 1)
    rotation_angle = vector_angle(plane_normal, z_axis)

    # Calculate rotation axis
    plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
    u1 = b / plane_normal_length
    u2 = -a / plane_normal_length
    rot_axis = (u1, u2, 0)
    # Generate axis-angle representation
    optim_factor = 1.4

    axis_angle = tuple([x * rotation_angle * optim_factor for x in rot_axis])

    # Rotate point cloud
    R = pcd.get_rotation_matrix_from_axis_angle(axis_angle)
    mesh.rotate(R, center=(0, 0, 0))

    obb = mesh.get_oriented_bounding_box()
    x_axis = np.array([1, 0, 0])
    bbox_angle = np.arccos(np.dot(obb.R[:, 0], x_axis))

    return rotation_angle * optim_factor, u1, 0, u2, bbox_angle * optim_factor


def init_scene(
    resolution: int = 1024,
    engine: str = "CYCLES",
    file_format: str = "OPEN_EXR",  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: int = "16",  # ('8', '16')
    color_mode: str = "RGB",  # ('RGB', 'RGBA', ...)
):
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
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
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
        map.size = [1.0]
        map.use_min = True
        map.min = [0]
        map.use_max = True
        map.max = [255]

        links.new(render_layers.outputs["Depth"], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = ""
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = file_format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    albedo_file_output.base_path = ""
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = file_format
    albedo_file_output.format.color_mode = color_mode
    albedo_file_output.format.color_depth = color_depth
    links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

    # Create id map output nodes
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = "ID Output"
    id_file_output.base_path = ""
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = file_format
    id_file_output.format.color_depth = color_depth

    if file_format == "OPEN_EXR":
        links.new(render_layers.outputs["IndexOB"], id_file_output.inputs[0])
    else:
        id_file_output.format.color_mode = "BW"

        divide_node = nodes.new(type="CompositorNodeMath")
        divide_node.operation = "DIVIDE"
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2 ** int(color_depth)

        links.new(render_layers.outputs["IndexOB"], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    return depth_file_output, id_file_output


def render_model(
    model_file: str,
    output_dir: str,
    depth_file_output, id_file_output,
    max_dim: int = 0.8,
    mesh_rotation_angle: tuple = (0, 0, 0, 0, 0),
):
    context = bpy.context
    scene = bpy.context.scene

    # Import textured mesh
    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.import_scene.obj(filepath=model_file)

    obj = bpy.context.selected_objects[0]

    context.view_layer.objects.active = obj

    # set object origin to center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="BOUNDS")

    # move object to 000
    bpy.ops.object.location_clear(clear_delta=False)
    bpy.ops.object.location_clear(clear_delta=True)
    bpy.context.object.location = (0, 0, 0)

    # align object
    rot_angle, x, y, z, bbox_angle = mesh_rotation_angle
    q = Quaternion((x, y, z), rot_angle)
    bpy.context.object.matrix_world = (
        q.to_matrix().to_4x4() @ bpy.context.object.matrix_world
    )
    q = Quaternion((1, 0, 0), math.radians(180))
    bpy.context.object.matrix_world = (
        q.to_matrix().to_4x4() @ bpy.context.object.matrix_world
    )
    q = Quaternion((0, 0, 0), bbox_angle)
    bpy.context.object.matrix_world = (
        q.to_matrix().to_4x4() @ bpy.context.object.matrix_world
    )

    # scale object
    max_dim_current = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    ratio = max_dim / max_dim_current
    bpy.ops.transform.resize(value=(ratio, ratio, ratio))
    bpy.ops.object.transform_apply(scale=True)

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes["Principled BSDF"]
        node.inputs["Specular"].default_value = 0.05

    # Remove double vertices to improve mesh quality
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Adds edge split filter
    bpy.ops.object.modifier_add(type="EDGE_SPLIT")
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Set objekt IDs
    obj.pass_index = 1

    # Make light just directional, disable shadows.
    light = bpy.data.lights["Light"]
    light.type = "SUN"
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 10.0

    # Add another light source so stuff facing away from light
    # is not completely dark
    bpy.ops.object.light_add(type="SUN")
    light2 = bpy.data.lights["Sun"]
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.015
    bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Light"].rotation_euler
    bpy.data.objects["Sun"].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects["Camera"]
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    image_fn = model_file.split("/")[-1].split(".")[0]
    image_fn = image_fn.replace(" ", "_").replace("-", "_")

    render_fp = f"{output_dir}/render/{image_fn}"
    depth_fp = f"{output_dir}/depth/{image_fn}"

    scene.render.filepath = render_fp
    depth_file_output.file_slots[0].path = depth_fp
    id_file_output.file_slots[0].path = render_fp + "_id"

    bpy.ops.render.render(write_still=True)  # render still
    context.active_object.select_set(True)
    for obj in bpy.context.scene.objects:
        # Check if the object is a mesh
        if obj.type == 'MESH':
            # If it is, delete it
            bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    # List of file paths to 3D models
    input_dir = "/Users/akonevskikh/Work/wwwwwork/cas/Paleontology/Meshes-2023-06"
    output_dir = "renders"
    depth_file_output, id_file_output = init_scene(resolution=2048)
    for model in sorted(glob.glob(f"{input_dir}/*.obj")):
        print(model)
        mesh_rot_angle = get_mesh_rotation(model)
        render_model(model, "renders", depth_file_output, id_file_output,
                     mesh_rotation_angle=mesh_rot_angle)
