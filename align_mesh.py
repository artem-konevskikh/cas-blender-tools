"""
Load a mesh, align it to the floor plane and render it

Rotation algorithm is from https://stackoverflow.com/questions/62596854/aligning-a-point-cloud-with-the-floor-plane-using-open3d
"""

import math
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def vector_angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_floor_plane(pcd, dist_threshold=0.02):
    plane_model, _ = pcd.segment_plane(distance_threshold=dist_threshold,
                                       ransac_n=3, num_iterations=1000)
    return plane_model


def rotate_mesh(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    floor = get_floor_plane(pcd)
    a, b, c, d = floor

    # Translate plane to coordinate center
    pcd.translate((0, -d/c, 0))

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


def render_mesh(mesh,
                width: int = 1024, height: int = 1024,
                filename: str = "render.png",
                depth_map_filename: str = "depth.png"):
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    mat = o3d.visualization.rendering.MaterialRecord()
    # mat.albedo_img = mesh.textures[1]
    # mat.aspect_ratio = 1.0
    mat.shader = "defaultLit"
    renderer_pc.scene.add_geometry("mesh", mesh, mat)

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 15.0  # between 5 and 90 degrees
    aspect_ratio = width / height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer_pc.scene.camera.set_projection(vertical_field_of_view,
                                            aspect_ratio, near_plane,
                                            far_plane, fov_type)

    # Look at the origin from the front (along the -Z direction,
    # into the screen), with Y as Up.
    center = [0, 0, 0]  # look_at target
    eye = [0, 0, 2]  # camera position
    up = [0, 1, 0]  # camera orientation
    renderer_pc.scene.camera.look_at(center, eye, up)

    depth_image = np.asarray(renderer_pc.render_to_depth_image())
    np.save('depth', depth_image)

    normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    plt.imshow(depth_image)
    plt.savefig(depth_map_filename)


input_folder = "models"
for f in sorted(glob.glob(f"{input_folder}/*.obj")):
    filename = f.split('/')[-1].split('.')[0].replace(' ', '_')
    print(filename)
    textured_mesh = o3d.io.read_triangle_mesh(f)
    textured_mesh.compute_vertex_normals()
    textured_mesh.textures = [textured_mesh.textures[1], textured_mesh.textures[1]]

    rotate_mesh(textured_mesh)
    render_mesh(textured_mesh, filename=f"renders/{filename}_render.png",
                depth_map_filename=f"renders/{filename}_depthmap.png")
