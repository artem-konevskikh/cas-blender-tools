import pyrender
import numpy as np
import trimesh
from PIL import Image


# Load the 3D model
mesh = trimesh.load('models/CAS-Kalksburg-1 Sharp fusion 1.obj')

# Create a Pyrender mesh with PBR material
material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(1.0, 1.0, 1.0, 1.0)
)
pr_mesh = pyrender.Mesh.from_trimesh(mesh)

# Create a scene and add the mesh
scene = pyrender.Scene()
scene.add(pr_mesh)

# Compute the centroid of the mesh
centroid = mesh.centroid

# Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

# Move the camera to be at a distance from the centroid of the mesh
camera_pose[0, 3] = centroid[0]
camera_pose[1, 3] = centroid[1]
camera_pose[2, 3] = centroid[2] + 2 * max(mesh.extents)

scene.add(camera, pose=camera_pose)


# Set up the light -- a single spot light in the same spot as the camera
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0)
scene.add(light, pose=camera_pose)

# Render the scene
renderer = pyrender.OffscreenRenderer(2048, 2048)
color, depth = renderer.render(scene)
# Convert the rendered color image to a PIL Image and save it
color_image = Image.fromarray((color).astype(np.uint8))
color_image.save('renders2/color.png')

# Convert the rendered depth map to a PIL Image and save it
depth_image = Image.fromarray((depth * 255 * 255).astype(np.uint16))
depth_image.save('renders2/depth.png')
