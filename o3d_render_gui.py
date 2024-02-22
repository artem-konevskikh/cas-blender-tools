"""
Load a mesh, align it to the floor plane and render it

Rotation algorithm is from https://stackoverflow.com/questions/62596854/
    aligning-a-point-cloud-with-the-floor-plane-using-open3d
"""

from datetime import datetime
import math
import glob
import copy
import open3d as o3d
import numpy as np
from PIL import Image


def vector_angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_floor_plane(pcd, dist_threshold=0.02):
    plane_model, _ = pcd.segment_plane(
        distance_threshold=dist_threshold, ransac_n=3, num_iterations=1000
    )
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


def custom_draw_geometry_with_key_callback(meshes, output_dir):

    custom_draw_geometry_with_key_callback.index = 0
    custom_draw_geometry_with_key_callback.render_name = ""
    custom_draw_geometry_with_key_callback.depth_name = ""

    def update_vis(vis, geometry):
        vis.clear_geometries()
        vis.add_geometry(geometry)
        vis.update_renderer()

    def get_filename(model_path):
        fname = model_path.split("/")[-1].split(".")[0]
        fname = fname.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_draw_geometry_with_key_callback.render_name = (
            f"{output_dir}/render_{fname}_{timestamp}.png"
        )
        custom_draw_geometry_with_key_callback.depth_name = (
            f"{output_dir}/depth_{fname}_{timestamp}.png"
        )

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_image_and_depth(vis):
        print("capture image")
        depth = vis.capture_depth_float_buffer()
        depth_image = np.asarray(depth)
        normalized_image = (depth_image - depth_image.min()) / (
            depth_image.max() - depth_image.min()
        )
        normalized_image = (normalized_image * 65535).astype(np.uint16)
        get_filename(meshes[custom_draw_geometry_with_key_callback.index])
        Image.fromarray(normalized_image).save(
            custom_draw_geometry_with_key_callback.depth_name
        )
        vis.capture_screen_image(
            custom_draw_geometry_with_key_callback.render_name
        )
        print("image saved")
        return False

    def next_model(vis):
        custom_draw_geometry_with_key_callback.index += 1
        custom_draw_geometry_with_key_callback.index %= len(meshes)
        mesh_path = meshes[custom_draw_geometry_with_key_callback.index]
        model = load_mesh(mesh_path)
        get_filename(mesh_path)
        update_vis(vis, model)

    def prev_model(vis):
        custom_draw_geometry_with_key_callback.index -= 1
        custom_draw_geometry_with_key_callback.index %= len(meshes)
        mesh_path = meshes[custom_draw_geometry_with_key_callback.index]
        model = load_mesh(mesh_path)
        get_filename(mesh_path)
        update_vis(vis, model)

    key_to_callback = {}
    key_to_callback[ord("B")] = change_background_to_black
    key_to_callback[ord("R")] = capture_image_and_depth
    key_to_callback[ord(",")] = prev_model
    key_to_callback[ord(".")] = next_model
    o3d.visualization.draw_geometries_with_key_callbacks(
        [load_mesh(meshes[0])], key_to_callback, width=2048, height=2048
    )


if __name__ == "__main__":
    # List of file paths to 3D models
    input_dir = "models"
    output_dir = "renders"
    meshes = sorted(glob.glob(f"{input_dir}/*.obj"))

    # Visualize models one by one with key callback
    custom_draw_geometry_with_key_callback(meshes, output_dir)
