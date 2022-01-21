#Importing the required libraries
import numpy as np
import cv2
try:
    import open3d

except ImportError:
    print("Please Install Open3D")
    exit()


# Converts 2D matrix to 3D point cloud
def convert_to_3d(left_image_matrix, depth_map_matrix):   # Depth map should be in Grayscale

    # Make the shape of the left image and depth map are same
    left_image_matrix = cv2.resize(left_image_matrix,  depth_map_matrix.shape[::-1])
    left_image_matrix = cv2.cvtColor(left_image_matrix, cv2.COLOR_BGR2RGB)

    height, width, _ = left_image_matrix.shape

    point_color = [] 
    # Stores Color Information of the original Image
    for row_pixel in range(height):
        for coloum_pixel in range(width):
            point_color.append(left_image_matrix[row_pixel][coloum_pixel])
    point_color = np.array(point_color)

    y_coordinates = []
    x_coordinates = []

    for row_pixel in range(height):
        for column_pixel in range(width):
            y_coordinates.append(column_pixel)
            x_coordinates.append(row_pixel)
 
    z_coordinates = depth_map_matrix.flatten()
    points = [] 

    for index in range(len(x_coordinates)):
        points.append([x_coordinates[index], y_coordinates[index], z_coordinates[index]])

    # Visualize the point Cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    point_cloud.colors = open3d.utility.Vector3dVector(point_color / 255.0)
    open3d.visualization.draw_geometries([point_cloud])
