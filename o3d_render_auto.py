"""
Load a mesh, align it to the floor plane and render it
Rotation algorithm is from https://stackoverflow.com/questions/62596854/aligning-a-point-cloud-with-the-floor-plane-using-open3d
"""

import copy
from datetime import datetime
import glob
import math
import open3d as o3d
import numpy as np
from PIL import Image


def vector_angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_floor_plane(pcd, dist_threshold=0.02):
    plane_model, _ = pcd.segment_plane(distance_threshold=dist_threshold,
                                       ransac_n=3, num_iterations=1000)
    return plane_model


def rotate_mesh(input_mesh):
    mesh = copy.deepcopy(input_mesh)
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
    return mesh


def load_mesh(mesh_path):
    model = o3d.io.read_triangle_mesh(mesh_path)
    model.compute_vertex_normals()
    model.textures = [model.textures[1], model.textures[1]]
    return rotate_mesh(model)


def get_filename(model_path, output_dir="renders"):
    fname = model_path.split("/")[-1].split(".")[0]
    fname = fname.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_name = f"{output_dir}/render_{fname}_{timestamp}.png"
    depth_name = f"{output_dir}/depth_{fname}_{timestamp}.png"
    return render_name, depth_name


def render_mesh(mesh,
                width: int = 1024, height: int = 1024,
                render_name: str = "render.png",
                depth_name: str = "depth.png"):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.thickness = 0.3
    material.transmission = 0.6

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()

    distance = 1.5*extent.max()
    
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    renderer_pc.scene.add_geometry("mesh", mesh, material)
    renderer_pc.scene.camera.set_projection(60, 1.0, 0.1, 1000.0, o3d.visualization.rendering.Camera.FovType.Horizontal)
    renderer_pc.scene.camera.look_at(center.tolist(), [100, 100, 100], [0, 0, distance])


    depth_image = np.asarray(renderer_pc.render_to_depth_image())
    minval = np.nanmin(depth_image)
    maxval = depth_image.max()
    normalized_image = (depth_image - minval) / maxval - minval
    normalized_image = (normalized_image * 65535).astype(np.uint16)
    normalized_image = 65535 - np.where(
        normalized_image == 0, 65535, normalized_image
    )
    min = np.min(normalized_image)  # result=144
    max = np.max(normalized_image)
    LUT = np.zeros(65535, dtype=np.uint16)
    LUT[min: max + 1] = np.linspace(
        start=0, stop=65535, num=(max - min) + 1,
        endpoint=True, dtype=np.uint16
    )
    normalized_image = LUT[normalized_image]
    Image.fromarray(normalized_image).save(depth_name)


# Load the mesh
if __name__ == "__main__":
    # List of file paths to 3D models
    input_dir = "models"
    output_dir = "renders"
    meshes = sorted(glob.glob(f"{input_dir}/*.obj"))
    for mesh_path in meshes:
        print(mesh_path)
        # mesh_path = "models/CAS-Kalksburg-1 Sharp fusion 1.obj"
        textured_mesh = o3d.io.read_triangle_mesh(mesh_path)
        textured_mesh.compute_vertex_normals()
        textured_mesh.textures = [textured_mesh.textures[1],
                                  textured_mesh.textures[1]]
        # textured_mesh = rotate_mesh(textured_mesh)
        render_name, depth_name = get_filename(mesh_path, output_dir)
        render_mesh(textured_mesh, 2048, 2048, render_name, depth_name)
