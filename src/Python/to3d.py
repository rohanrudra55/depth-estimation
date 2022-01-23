"""MIT License

Copyright (c) 2022 Rohan Rudra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
#Importing the required libraries
import numpy as np
import cv2
import open3d


# Converts 2D matrix to 3D point cloud
def convert_to_3d(left_image_matrix, depth_map_matrix):   # Depth map should be in Grayscale

    # Make the shape of the left image and depth map are same
    left_image_matrix = cv2.resize(left_image_matrix,  depth_map_matrix.shape[::-1])
    left_image_matrix = cv2.cvtColor(left_image_matrix, cv2.COLOR_BGR2RGB)

    height, width, depth = left_image_matrix.shape

    point_color = np.array(left_image_matrix.reshape((height*width), depth)) 
    z_coordinates = depth_map_matrix.flatten()
    points = [] 
    index=0
    # Stores Color Information of the original Image
    for row_pixel in range(height):
        for coloum_pixel in range(width):
            points.append([row_pixel, coloum_pixel, z_coordinates[index]])
            index+=1
            

    # Visualize the point Cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    point_cloud.colors = open3d.utility.Vector3dVector(point_color / 255.0)
    open3d.visualization.draw_geometries([point_cloud])
