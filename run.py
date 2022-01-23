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

from DE.sensors.camera import fetch
from DE.processing.camera import to3d
from DE.processing.camera import depth
import numpy as np
import cv2
import argparse
import yaml



parser=argparse.ArgumentParser(description="Calculate the depth map of an image and get a 3D model of it.\nThe images should be a stereo pair.")
parser.add_argument('-s','--save',type=bool,default=False,help='To save the results')
parser.add_argument('-3d','--vis3D',type=bool,default=False,help='To compute the 3D from the depth map pass 0')
parser.add_argument('-i','--indian',type=bool,default=False,help='To use indian dataset')
parser.add_argument('-n','--img_num',type=int,default=0)
args=parser.parse_args()

with open('DE/config.yaml') as config:
    data = yaml.load(config, Loader=yaml.FullLoader)
    left_frame = fetch.Camera(data['LEFT_IMAGE'],True,data['EXTENSION'])
    right_frame = fetch.Camera(data['RIGHT_IMAGE'],True,data['EXTENSION'])
    p_0 = np.array(data['P0'],dtype=np.float64)
    p_1 = np.array(data['P1'],dtype=np.float64)

    pro=depth.Estimator(enhance=True,leftProjMat=p_0,rightProjMat=p_1)
    pcl=to3d.Visualize()
    
    nxt_L = nxt_R = True
    index=0

    while (nxt_L and nxt_R) :
        l_img, nxt_L = left_frame.get()
        r_img, nxt_R = right_frame.get()

        if nxt_L and nxt_R and index == args.img_num:
            d_img=pro.estimate(r_img,l_img)
            pcl.convert_to_3d(l_img,d_img)

        index+=1