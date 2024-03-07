import os
from math import radians
from pathlib import Path
from typing import Dict, List
import bpy
import json


def set_render_settings(
    resolution_x=4096,
    resolution_y=4096,
    file_format: str = "PNG",  # ('PNG', 'OPEN_EXR', 'JPEG')
    color_depth: str = "16",  # ('8', '16')
    color_mode: str = "RGBA",  # ('RGB', 'RGBA', ...)
):
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


def load_material(material_meta: dict):
    mat = bpy.data.materials.new(name=material_meta["name"])
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links

    # Reuse the material output node that is created by default
    material_output = mat_nodes.get("Material Output")
    # Create the Principled BSDF node
    bsdf = mat_nodes["Principled BSDF"]
    mat_links.new(material_output.inputs["Surface"], bsdf.outputs["BSDF"])
    # Create the Texture Coordinate node
    mapping_node = mat_nodes.new("ShaderNodeMapping")
    tex_coordinate = mat_nodes.new("ShaderNodeTexCoord")
    mat_links.new(mapping_node.inputs["Vector"], tex_coordinate.outputs["UV"])

    # Create the Image Texture node
    tex_image = mat_nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(material_meta["albedo"])
    mat_links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])
    mat_links.new(tex_image.inputs["Vector"], mapping_node.outputs["Vector"])

    # Create Image Texture node and load the displacement texture.
    # You need to add the actual path to the texture.
    displacement_tex = mat_nodes.new("ShaderNodeTexImage")
    displacement_tex.image = bpy.data.images.load(
        material_meta["displacement"]
    )
    displacement_tex.image.colorspace_settings.name = "Non-Color"

    # Create the Displacement node
    displacement = mat_nodes.new("ShaderNodeDisplacement")
    # displacement.inputs["Scale"].default_value = 0.01

    # Connect the Texture Coordinate node to the displacement texture.
    # This uses the active UV map of the object.
    mat_links.new(
        displacement_tex.inputs["Vector"], mapping_node.outputs["Vector"]
    )

    # Connect the displacement texture to the Displacement node
    mat_links.new(
        displacement.inputs["Height"], displacement_tex.outputs["Color"]
    )

    # Connect the Displacement node to the Material Output node
    mat_links.new(
        material_output.inputs["Displacement"],
        displacement.outputs["Displacement"]
    )

    # Create the Normal texture node
    normal_tex = mat_nodes.new("ShaderNodeTexImage")
    normal_tex.image = bpy.data.images.load(material_meta["normal"])
    normal_tex.image.colorspace_settings.name = "Non-Color"
    normal_map = mat_nodes.new("ShaderNodeNormalMap")
    # normal_map.inputs['Strength'].default_value = 0.5
    mat_links.new(normal_map.inputs["Color"], normal_tex.outputs["Color"])
    mat_links.new(bsdf.inputs["Normal"], normal_map.outputs["Normal"])
    mat_links.new(normal_tex.inputs["Vector"], mapping_node.outputs["Vector"])

    # Create the Normal texture node
    roughness_tex = mat_nodes.new("ShaderNodeTexImage")
    roughness_tex.image = bpy.data.images.load(material_meta["roughness"])
    roughness_tex.image.colorspace_settings.name = "Non-Color"
    mat_links.new(bsdf.inputs["Roughness"], roughness_tex.outputs["Color"])
    mat_links.new(
        roughness_tex.inputs["Vector"], mapping_node.outputs["Vector"]
    )

    return mat


def load_mesh_with_material(model_file: str, material, max_dim: float = 0.5):
    bpy.ops.import_scene.obj(filepath=str(model_file))
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler[0] = radians(3)
    obj.rotation_euler[1] = radians(0.2)

    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.subdivision_set(level=3)

    # set object origin to center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="BOUNDS")

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
    bpy.ops.mesh.select_all(action="SELECT")  # for all faces
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
    bpy.ops.object.editmode_toggle()
    # obj.select_set(False)

    # scale object
    max_dim_current = max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    ratio = max_dim / max_dim_current
    bpy.ops.transform.resize(value=(ratio, ratio, ratio))
    bpy.ops.object.transform_apply(scale=True)

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes["Principled BSDF"]
        node.inputs["Specular"].default_value = 0.0001

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
    output_dir,
    rotation_steps=32,
    subject=bpy.context.object,
    depth_file_output=None,
):
    original_rotation = subject.rotation_euler
    for step in range(0, rotation_steps):
        subject.rotation_euler[2] = radians(step * (360.0 / rotation_steps))
        render_fn = os.path.join(output_dir, f"render{step}.jpg")
        depth_fn = os.path.join(output_dir, f"depth{step}")
        bpy.context.scene.render.filepath = os.path.join(render_fn)
        if depth_file_output:
            depth_file_output.base_path = os.path.join(depth_fn)
        bpy.ops.render.render(write_still=True)
    subject.rotation_euler = original_rotation


def rotate_and_render(
    output_file,
    angle=1,
    subject=bpy.context.object,
    depth_file_output=None,
):
    subject.rotation_euler[2] = radians(angle)
    bpy.context.scene.render.filepath = f"{output_file}_render"
    if depth_file_output:
        depth_file_output.base_path = f"{output_file}_depth"
    bpy.ops.render.render(write_still=True)


def sort_models_by_name(models):
    return list(sorted(models, key=lambda x: int(x.stem.split("_")[-1])))


def render_interpolation(
    input_dir: str,
    output_dir: str,
    resolution: int,
    materials: List[Dict],
    loop: bool = False,
):
    models = Path(input_dir).rglob("*.obj")
    models = sort_models_by_name(models)
    if loop:
        models += models[::-1]
    for mat in materials:
        for i, model in enumerate(models):
            if i == 0 or i % 5 != 0 or i % 10 == 0:
                continue
            depth_file_output = set_render_settings(resolution, resolution)
            obj = load_mesh_with_material(
                str(model), load_material(mat), max_dim=0.8
            )
            rotate_and_render(
                f"{output_dir}/{mat['name']}/frame_{i:08d}",
                angle=i,
                subject=obj,
                # depth_file_output=depth_file_output,
            )
            for obj in bpy.context.scene.objects:
                # Check if the object is a mesh
                if obj.type == 'MESH':
                    # If it is, delete it
                    bpy.data.objects.remove(obj, do_unlink=True)


def render_one_frame(
    model: str,
    output_file: str,
    resolution: int,
    material: Dict,
):
    depth_file_output = set_render_settings(resolution, resolution)
    obj = load_mesh_with_material(
        model, load_material(material), max_dim=0.8
    )
    rotate_and_render(
        output_file,
        angle=0,
        subject=obj,
    )


def render_folder(
    input_dir: str,
    output_dir: str,
    resolution: int,
    materials: List[Dict],
):
    models = Path(input_dir).rglob("*.obj")
    for mat in materials:
        for i, model in enumerate(models):
            depth_file_output = set_render_settings(resolution, resolution)
            obj = load_mesh_with_material(
                str(model), load_material(mat), max_dim=0.8
            )
            render_one_frame(
                str(model),
                f"{output_dir}/{mat['name']}/{model.stem}",
                resolution,
                mat,
            )
            for obj in bpy.context.scene.objects:
                # Check if the object is a mesh
                if obj.type == 'MESH':
                    # If it is, delete it
                    bpy.data.objects.remove(obj, do_unlink=True)

stone_material = {
    "name": "stone_floor_tkkkeicew",
    "albedo": "materials/stone_floor_tkkkeicew/tkkkeicew_8K_Albedo.jpg",
    "displacement": "materials/stone_floor_tkkkeicew/tkkkeicew_8K_Displacement.jpg",
    "normal": "materials/stone_floor_tkkkeicew/tkkkeicew_8K_Normal.jpg",
    "roughness": "materials/stone_floor_tkkkeicew/tkkkeicew_8K_Roughness.jpg",
}
wall_material = {
    "name": "Wall_Painted_rlvklup0_4K_surface_ms",
    "albedo": "materials/Wall_Painted_rlvklup0_4K_surface_ms/rlvklup_4K_Albedo.jpg",
    "displacement": "materials/Wall_Painted_rlvklup0_4K_surface_ms/rlvklup_4K_Displacement.jpg",
    "normal": "materials/Wall_Painted_rlvklup0_4K_surface_ms/rlvklup_4K_Normal.jpg",
    "roughness": "materials/Wall_Painted_rlvklup0_4K_surface_ms/rlvklup_4K_Roughness.jpg",

}
concrete_material = {
    "name": "Concrete_Damaged_tefmajbn_8K_surface_ms",
    "albedo": "materials/Concrete_Damaged_tefmajbn_8K_surface_ms/tefmajbn_8K_Albedo.jpg",
    "displacement": "materials/Concrete_Damaged_tefmajbn_8K_surface_ms/tefmajbn_8K_Displacement.jpg",
    "normal": "materials/Concrete_Damaged_tefmajbn_8K_surface_ms/tefmajbn_8K_Normal.jpg",
    "roughness": "materials/Concrete_Damaged_tefmajbn_8K_surface_ms/tefmajbn_8K_Roughness.jpg",
}

plaster_material = {
    "name": "Plaster_Damaged_pjBji0_4K_surface_ms",
    "albedo": "materials/Plaster_Damaged_pjBji0_4K_surface_ms/pjBji_4K_Albedo.jpg",
    "displacement": "materials/Plaster_Damaged_pjBji0_4K_surface_ms/pjBji_4K_Displacement.jpg",
    "normal": "materials/Plaster_Damaged_pjBji0_4K_surface_ms/pjBji_4K_Normal.jpg",
    "roughness": "materials/Plaster_Damaged_pjBji0_4K_surface_ms/pjBji_4K_Roughness.jpg",
}

# depth_file_output = set_render_settings(1024, 1024)
# # obj = load_textured_mesh(
# #     'models/CAS-Kalksburg-1 Sharp fusion 1.obj',
# #     max_dim=0.8
# # )
# stone_mat = load_material(stone_material)
# obj = load_mesh_with_material(
#     '/Users/akonevskikh/Work/wwwwwork/cas/Paleontology/interpolations/interpolations_feb24/skulls01/interpolate_mesh_runs_skulls-last-v2.ckpt_True_0.001_128_1.0_0.obj',
#     stone_mat,
#     max_dim=0.8
# )
# rotate_and_render(
#     "/Users/akonevskikh/Work/wwwwwork/cas/Paleontology/renders-test/02",
#     subject=obj,
#     # depth_file_output=depth_file_output,
# )

# input_dir = "/home/aicu/ai/3d/sdf-stylegan/outputs/skulls005"
# output_dir = "renders/skulls005"
# render_interpolation(
#     input_dir,
#     output_dir,
#     2048,
#     [stone_material, wall_material, concrete_material, plaster_material],
#     loop=True,
# )
# # Load materials from materials.json
# with open('materials.json') as f:
#     materials = json.load(f)


# for material in materials:
#     if os.path.exists(f"renders/skulls005/test/{material['name']}_00000000_render.jpg"):
#         continue
#     render_one_frame(
#         "../interpolations/interpolations_feb24/skulls01/interpolate_mesh_runs_skulls-last-v2.ckpt_True_0.001_128_1.0_0.obj",
#         f"renders/skulls005/test2/{material['name']}_00000000",
#         2048,
#         material,
#     )


input_dir = "../interpolations/interpolations_feb24/skulls01/"
output_dir = "renders/skulls01-5"
render_interpolation(
    input_dir,
    output_dir,
    2048,
    [concrete_material],
    loop=True,
)