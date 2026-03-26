import cv2
import numpy as np
import trimesh
import torch

# Load MiDaS model for depth estimation
midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', force_reload=True)
midas_model.eval()

# Load the input image
image_path = 'input_image.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Depth estimation
with torch.no_grad():
    depth_map = midas_model.forward(image)[0]

depth_map = depth_map.squeeze().cpu().numpy()

# Mesh generation
mesh = trimesh.Trimesh()
mesh.vertices = np.array([[x, y, depth_map[y][x]] for y in range(height) for x in range(width)])
faces = []
for y in range(height-1):
    for x in range(width-1):
        faces.append([y * width + x, y * width + (x + 1), (y + 1) * width + x])
        faces.append([(y + 1) * width + x, y * width + (x + 1), (y + 1) * width + (x + 1)])
mesh.faces = np.array(faces)

# Export to OBJ format
mesh.export('output_model.obj')
# Export to GLB format
mesh.export('output_model.glb')

print('3D model created successfully!')
