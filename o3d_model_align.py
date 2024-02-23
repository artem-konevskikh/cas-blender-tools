"""
Load a mesh, align it to the floor plane and render it
Rotation algorithm is from https://stackoverflow.com/questions/62596854/aligning-a-point-cloud-with-the-floor-plane-using-open3d
"""

import copy
import glob
import math
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
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

for mesh_path in glob.glob("models/*.obj"):
    fname = mesh_path.split("/")[-1]
    print(fname)
    textured_mesh = load_mesh(mesh_path)
    o3d.io.write_triangle_mesh(f"models-aligned/{fname}",
                               textured_mesh,
                               compressed=True,
                               write_vertex_normals=False,
                               print_progress=True)
print("done")
