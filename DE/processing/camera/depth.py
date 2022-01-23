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

import numpy as np
import cv2
import matplotlib.pyplot as plt


class Estimator:
    def __init__(self,enhance=False,leftProjMat=[],rightProjMat=[]):
        """
        to_enchance = To enchange images before processing
        left_C_mat = Left camera matrix 
        left_T_vec = Left camera translation vector 
        right_T_vec = Right camera translation vector 
        """
        self.to_enhance = enhance

        # Decomposing the Projection Matrixs
        self.left_C_mat, _,self.left_T_vec, _, _, _, _ = cv2.decomposeProjectionMatrix(leftProjMat)
        _, _,self.right_T_vec, _, _, _, _ = cv2.decomposeProjectionMatrix(rightProjMat)
        

    def enhance(self,image):
        """
            To preprocess the image with differnt algorithms
                # Gaussian Blur to remove the noise
                # Gamma Equilization 
                # Histogram filter to equilize the lights
        """
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        smoothen_image = cv2.GaussianBlur(image,(5,5),0.1)
        
        gamma=0.75
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        gamma_equilized_image = cv2.LUT(smoothen_image, lookUpTable)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        histogram_filtered_image = clahe.apply(gamma_equilized_image)

        return histogram_filtered_image
    
    def estimate(self,rightimage ,leftimage,depth=False):
        """
        Using Stereo SGBM to calculate disparity map and 
        traditional approach to calculate depth

        Input:
                leftimage = Left image matrix
                rightimage = Right image matrix
                depth = To calculate depth 
        Output:
                Disparity map and depth map of the input image
        """

        if self.to_enhance:
            leftimage = self.enhance(leftimage)
            rightimage = self.enhance(rightimage)

        # Calculating Depth Map using Stereo SGBM
        stereo = cv2.StereoSGBM_create(numDisparities=16*6, 
                                        blockSize=15,P1=4000,P2=15000,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disparity_map = stereo.compute(leftimage,rightimage)

        if depth:
            disparity = disparity_map

            f = self.left_C_mat[0][0]
            b = self.right_T_vec[0] - self.left_T_vec[0]

            # Avoid instability and division by zero
            disparity_map[disparity_map == 0.0] = 0.1
            disparity_map[disparity_map == -1.0] = 0.1
        
            depth_map = np.ones(disparity_map.shape)
            depth_map = f * b / (disparity_map+0.00000001)

            return disparity,depth_map

        return disparity_map[:,95:]