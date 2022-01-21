#Importing libraries
try:
    import argparse
    import numpy as np
    import pandas as pd
    import cv2
    import operations as op
except ImportError:
    print("Some files of the program missing or misplaced")
    exit()

#The main function:
def run():
    parser=argparse.ArgumentParser(description="Calculate the depth map of an image and get a 3D model of it.\nThe images should be a stereo pair.")
    parser.add_argument('-l','--left_image',type=str, default='../../resource/left/000004.png',help='The image from the left camera.')
    parser.add_argument('-r','--right_image',type=str,default='../../resource/right/000004.png',help='The image from thr right camera.')
    parser.add_argument('-s','--show_result',type=bool,default=False,help='To show the different results')
    parser.add_argument('-3d','--To3D',type=bool,default=False,help='To compute the 3D from the depth map pass 0.')
    parser.add_argument('-d','--indian',type=bool,default=False,help='To use indian dataset')
    args=parser.parse_args()

    #Reading Images
    right_image_read = cv2.imread(args.right_image,0)
    left_image_read = cv2.imread(args.left_image,0)
    op.output(left_image_read,"Imported_Image",show=args.show_result)

    if args.indian:
        left_image_read,right_image_read=op.calib(left_image_read,right_image_read)

    #Pre-Processing
    processed_right_image,_,_=op.enhance(right_image_read)
    processed_left_image,_,_=op.enhance(left_image_read)
    op.output(processed_left_image,"Processed_Image",show=args.show_result)

    #Calculating the depth map and disparity map
    disparity_map,depth_map=op.estimateDepth(processed_right_image,processed_left_image,save=True)
    op.output(depth_map,"Depth Map",show=args.show_result,save=True)
    op.output(disparity_map,"Disparity Map",show=True,save=True)
    # 3D print(disparity_map)
    if args.To3D == True:
        import to3d #Importing the 3D convertion file
        left_image_colour=cv2.imread(args.left_image)
        to3d.convert_to_3d(left_image_colour,disparity_map[:,95:])

if __name__ == "__main__":
    #Running Program
    run()
    exit()#To terminate the program