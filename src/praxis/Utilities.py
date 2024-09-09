import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from praxis.Camera import MonocularCamera
from praxis.DepthEstimator import GLPNDepthEstimator


def calculate_intersection(hand_index_2D, pointing_unit_vector, objDetection, depth_map, cls_index=0):

    print("(================= Calculate_intersection")
    print(f"==>>> hand_index_2D={hand_index_2D}")
    print(f"==>>> pointing_unit_vector={pointing_unit_vector}")
    print(f"==>>> objDetection={objDetection}")

    if hand_index_2D[1] < 0: # incorrect hand coord
        print(f"incorrect hand coord: {hand_index_2D}")
        return 0,  objDetection.cls[cls_index]

    hand_index_3D = convert_2d_to_3d(depth_map, hand_index_2D)
    print(f"==>>> hand_index_3D={hand_index_3D}")

    x_min, y_min, x_max, y_max = objDetection.data[cls_index, :4]
    object_center_2D = torch.tensor([(x_min+x_max)/2, (y_min+y_max)/2])
    object_center_3D = convert_2d_to_3d(depth_map, object_center_2D)
    print(f"==>>> object_center_3D={object_center_3D}")

    vector_to_object = object_center_3D - hand_index_3D

    #print("================###############################################################")
    #print(f"Norm of DeePoint vector: {torch.norm(pointing_unit_vector).item()}")

    #draw_pointing_unit_vector = torch.tensor([pointing_unit_vector[0],pointing_unit_vector[1],-pointing_unit_vector[2]])
    #draw_vector_to_object = torch.tensor([vector_to_object[0], vector_to_object[1], -vector_to_object[2]])

    # Step 3: Normalize vector
    normalized_pointing_unit_vector = pointing_unit_vector.detach() / torch.norm(pointing_unit_vector.detach())
    normalized_vector_to_object = vector_to_object / torch.norm(vector_to_object)

    #print(f"Norm of object vector: {torch.norm(normalized_vector_to_object).item()}")
    #print(f"Norm of pointing vector: {torch.norm(normalized_pointing_unit_vector).item()}")

    # Plot both vectors
    #plot_3d_vector(hand_index_3D.numpy(), normalized_pointing_unit_vector.numpy(), color='r', label='Pointing Direction')
    #plot_3d_vector(hand_index_3D.numpy(), normalized_vector_to_object.numpy(), color='g', label='Vector to Object')

    #cosine_similarity = torch.clamp(torch.dot(pointing_unit_vector.detach(), normalized_vector_to_object), -1.0, 1.0)
    cosine_similarity = torch.dot(normalized_pointing_unit_vector, normalized_vector_to_object)

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


# Function to plot a 3D vector
def plot_3d_vector(origin, vector, color='b', label='Vector'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vector starting from the origin
    ax.quiver(origin[0], origin[1], origin[2],
              vector[0], vector[1], vector[2],
              length=1, color=color, label=label)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vector Visualization')

    # Set axis limits for better visualization
    axis_limit = 2
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])

    ax.legend()
    plt.show()




