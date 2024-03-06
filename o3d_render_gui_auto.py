import copy
from datetime import datetime
import glob
import math
import open3d as o3d
import time
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


def draw_geometries(geometries, output_dir="renders", delay=5):
    vis = o3d.visualization.Visualizer()

    def get_filename(model_path, output_dir="renders"):
        fname = model_path.split("/")[-1].split(".")[0]
        fname = fname.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_name = f"{output_dir}/render_{fname}_{timestamp}.png"
        depth_name = f"{output_dir}/depth_{fname}_{timestamp}.png"
        depth_norm_name = f"{output_dir}/depth_norm_{fname}_{timestamp}.png"
        return render_name, depth_name, depth_norm_name

    def load_geometry(mesh_path):
        model = o3d.io.read_triangle_mesh(mesh_path)
        model.compute_vertex_normals()
        model.textures = [model.textures[1], model.textures[1]]
        geometry = rotate_mesh(model)
        vis.clear_geometries()
        vis.add_geometry(geometry)

    vis.create_window(width=1920, height=1920)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for i, geometry in enumerate(geometries):
        load_geometry(geometry)
        render_fn, depth_fn, depth_norm_fn = get_filename(geometry, output_dir)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)  # delay in seconds

        # Capture the screen and depth buffer
        depth = vis.capture_depth_float_buffer(do_render=True)

        # Convert the depth buffer to a depth image
        depth_image = np.asarray(depth)
        minval = np.nanmin(depth_image)
        maxval = depth_image.max()
        normalized_image = (depth_image - minval) / maxval - minval
        normalized_image = (normalized_image * 65535).astype(np.uint16)
        normalized_image = 65535 - np.where(
            normalized_image == 0, 65535, normalized_image
        )
        Image.fromarray(normalized_image).save(
            depth_fn
        )
        min = np.min(normalized_image)  # result=144
        max = np.max(normalized_image)
        LUT = np.zeros(65535, dtype=np.uint16)
        LUT[min: max + 1] = np.linspace(
            start=0, stop=65535, num=(max - min) + 1,
            endpoint=True, dtype=np.uint16
        )
        normalized_image = LUT[normalized_image]

        Image.fromarray(normalized_image).save(
            depth_norm_fn
        )
        vis.capture_screen_image(render_fn)

    vis.destroy_window()


if __name__ == "__main__":
    # List of file paths to 3D models
    input_dir = "/Users/akonevskikh/Work/wwwwwork/cas/Paleontology/Meshes-2023-05"
    output_dir = "renders"
    meshes = sorted(glob.glob(f"{input_dir}/*.obj"))
    draw_geometries(meshes)
