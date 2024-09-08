import torch
from praxis.Camera import MonocularCamera
from praxis.DepthEstimator import GLPNDepthEstimator


def calculate_intersection(hand_index_2D, pointing_direction, objDetection, depth_map, cls_index=0):

    print("(================= Calculate_intersection")
    print(f"==>>> hand_index_2D={hand_index_2D}")
    print(f"==>>> pointing_direction={pointing_direction}")
    print(f"==>>> objDetection={objDetection}")


    hand_index_3D = convert_2d_to_3d(depth_map, hand_index_2D)
    print(f"==>>> hand_index_3D={hand_index_3D}")

    x_min, y_min, x_max, y_max = objDetection.data[cls_index, :4]
    object_center_2D = torch.tensor([(x_min+x_max)/2, (y_min+y_max)/2])
    object_center_3D = convert_2d_to_3d(depth_map, object_center_2D)
    print(f"==>>> object_center_3D={object_center_3D}")

    vector_to_object = object_center_3D - hand_index_3D

    # Step 3: Normalize both vectors
    normalized_pointing_direction = pointing_direction.detach() / torch.norm(pointing_direction)
    normalized_vector_to_object = vector_to_object / torch.norm(vector_to_object)

    cosine_similarity = torch.dot(normalized_pointing_direction, normalized_vector_to_object)

    print(f">>>>>>>>> Cosine Similarity: {cosine_similarity.item()}")
    print(f">>>>>>>>>================================================ Cosine Similarity: {cosine_similarity.item()}")
    print("================) end calculate_intersection")
    return cosine_similarity, objDetection.cls[cls_index]


def convert_2d_to_3d(depth_map, pixel_2d):

    print(depth_map)
    depth_value = depth_map[0][int(pixel_2d[1]), int(pixel_2d[0])]  # Z value in 3D
    camera = MonocularCamera()

    # Step 2: Back-project the 2D point to 3D coordinates
    X = (pixel_2d[0] - camera.c_x) * depth_value / camera.f_x  # X in 3D
    Y = (pixel_2d[1] - camera.c_y) * depth_value / camera.f_y  # Y in 3D
    Z = depth_value  # Z is the depth value

    point_3D = torch.tensor([X, Y, Z])
    return point_3D
