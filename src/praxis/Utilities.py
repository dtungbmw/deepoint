import torch
from praxis.Camera import MonocularCamera
from praxis.DepthEstimator import GLPNDepthEstimator

def calculate_intersection(joints, directionPred, objDetection, depthEstimation):

    print("==== calculate_intersection")
    #print(f"==>>> joints={joints}")
    print(f"==>>> direction={directionPred}")
    print(f"==>>> objDetection={objDetection[0].boxes}")
    print("==== end calculate_intersection")
    return True

def convert_2d_to_3d(video_file, pixel_2d):
    camera = MonocularCamera()
    image = camera.extractImage(video_file)
    gLPNDepthEstimator = GLPNDepthEstimator()
    depth_map = gLPNDepthEstimator.predict(image)
    print(depth_map.shape)
    print(depth_map)

    depth_value = depth_map[0][int(pixel_2d[1]), int(pixel_2d[0])]  # Z value in 3D

    # Step 2: Back-project the 2D point to 3D coordinates
    X = (pixel_2d[0] - camera.c_x) * depth_value / camera.f_x  # X in 3D
    Y = (pixel_2d[1] - camera.c_y) * depth_value / camera.f_y  # Y in 3D
    Z = depth_value  # Z is the depth value

    point_3d = torch.tensor([X, Y, Z])
    return point_3d