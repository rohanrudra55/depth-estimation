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
# It contains the differnt test functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import parameters as pmt

def output(image ,caption = "Image" ,save = False ,show = False):
    # To display the image and also to save the result
    if save == True:
        path="../"+caption+".jpg"
        cv2.imwrite(path,image) 
    if show == True:
        cv2.imshow(caption,image) 
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

def calib(leftimage,rightimage):

    lk, lr,lt, _, _, _, _ = cv2.decomposeProjectionMatrix(pmt.leftP)
    rk, rr,rt, _, _, _, _ = cv2.decomposeProjectionMatrix(pmt.rightP)

    r1,r2,p1,p2,Q,_,_=cv2.stereoRectify(lk ,pmt.leftDis ,lr ,pmt.rightDis ,pmt.imageSize ,pmt.r ,pmt.t)
    leftMap1,leftMap2=cv2.initUndistortRectifyMap(lk ,pmt.leftDis ,r1 ,p1 ,pmt.imageSize,cv2.CV_32FC1)
    rightMap1,rightMap2=cv2.initUndistortRectifyMap(rk ,pmt.rightDis ,r2 ,p2 ,pmt.imageSize,cv2.CV_32FC1)

    x=406
    y=248
    h=607
    w=1103

    leftimage=cv2.remap(leftimage,leftMap1,leftMap2,cv2.INTER_NEAREST)
    rightimage=cv2.remap(rightimage,rightMap1,rightMap2,cv2.INTER_NEAREST)
    leftimageCroped = leftimage[y:y+h, x:x+w]
    rightimageCroped= rightimage[y:y+h, x:x+w]

    return leftimageCroped,rightimageCroped

def enhance(image):
    # To preprocess the image with differnt algorithms
    # Gaussian Blur to remove the noise
    # Gamma Equilization 
    # Histogram filter to equilize the lights

    smoothen_image = cv2.GaussianBlur(image,(5,5),0.1)
    gamma=0.75
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_equilized_image = cv2.LUT(smoothen_image, lookUpTable)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    histogram_filtered_image = clahe.apply(gamma_equilized_image)

    return histogram_filtered_image,gamma_equilized_image,smoothen_image


def estimateDepth(right_image ,left_image ,save = False):

    # Decomposing the Projection Matrixs
    left_camera_matrix, _,left_translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(pmt.projMatL)
    _, _,right_translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(pmt.projMatR)

    # Calculating Depth Map Using Stereo BM
    # stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)

    # Calculating Depth Map using Stereo SGBM
    stereo = cv2.StereoSGBM_create(numDisparities=16*6, blockSize=15,P1=4000,P2=15000,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity_map = stereo.compute(left_image,right_image)
    disp=disparity_map

    f = left_camera_matrix[0][0]
    b=right_translation_vector[0] - left_translation_vector[0]

    # Avoid instability and division by zero
    disparity_map[disparity_map == 0.0] = 0.1
    disparity_map[disparity_map == -1.0] = 0.1
   
    depth_map = np.ones(disparity_map.shape)
    depth_map = f * b / (disparity_map+0.00000001)

    return disp,depth_map
    
