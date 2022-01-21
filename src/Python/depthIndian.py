import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import to3d

def display(image,tag="Image",rate=0):
    cv.imshow(tag,image)
    if cv.waitKey(rate)==27:
        cv.destroyAllWindows()
        print("System Closing...")
        exit(0)

def visualise2D(image,keypoints):
    radius = 2
    visualize=cv.cvtColor(image,cv.COLOR_GRAY2BGR,3)
    for point in keypoints:
        cv.circle(visualize,(int(point[0]),int(point[1])),radius=2,color=(0,255,0))
    return visualize

def limitFeatures(features,limit=0):
    total=len(features)
    if total >limit:
        features=features[:limit]
    # print("Reduced features form",total,"to","less than 1000")
    return features

def detectFeature(leftimage,rightimage):
    lk_params = dict( winSize = (21, 21),maxLevel = 3,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,30, 0.01))
    fast = cv.FastFeatureDetector_create()
    keypoint = fast.detect(leftimage,None)
    leftpoints = cv.KeyPoint_convert(keypoint)
    #Limiting features to certain points
    # leftpoints=limitFeatures(leftpoints,1000)
    #Optical flow in right image
    rightpoints,prvRst,err0 = cv.calcOpticalFlowPyrLK(leftimage,rightimage,leftpoints,None,**lk_params)
    return leftpoints,rightpoints

class camera:
    def __init__(self):
        #INDIAN DATASET
        imageSize=(1920,1080)
    
        leftDis=np.array([-0.177288, 0.0299518, 0.0, 0.0, 0.0])
        leftR= [[0.9999121806728741, -0.0008263015096623995, 0.013226797338454508], [0.0007244497743564737, 0.9999700663536536, 0.007703347921135117], [-0.013232766700197374, -0.007693089267956763, 0.9998828482692227]]
        leftP=np.array([[1403.11, 0.0, 1000.3, 0.0], [0.0, 1403.11, 549.43, 0.0], [0.0, 0.0, 1.0, 0.0]])

        rightDis=np.array([-0.173732, 0.0278646, 0.0, 0.0, 0.0])
        rightR=[[0.9999121806728741, -0.0008263015096623995, 0.013226797338454508], [0.0007244497743564737, 0.9999700663536536, 0.007703347921135117], [-0.013232766700197374, -0.007693089267956763, 0.9998828482692227]]
        rightP=np.array([[1401.92, 0.0, 1005.05, -168.37319999999997], [0.0, 1401.92, 620.367, 0.0], [0.0, 0.0, 1.0, 0.0]])

        t=np.array([-0.1200, 0.0, 0.0])
        r=np.array([[0.9999121806728741,-0.0008263015096623995, 0.013226797338454508],[0.0007244497743564737,    0.9999700663536536, 0.007703347921135117],[-0.013232766700197374, -0.007693089267956763,   0.9998828482692227]])

        lk, lr,lt, _, _, _, _ = cv.decomposeProjectionMatrix(leftP)
        rk, rr,rt, _, _, _, _ = cv.decomposeProjectionMatrix(rightP)

        self.flength = lk[0][0]
        self.baseline = rt[0] - lt[0]

        #Rectification
        r1,r2,p1,p2,Q,_,_=cv.stereoRectify(lk,leftDis,lr,rightDis,imageSize,r,t)
        self.leftMap1,self.leftMap2=cv.initUndistortRectifyMap(lk,leftDis,r1,p1,imageSize,cv.CV_32FC1)
        self.rightMap1,self.rightMap2=cv.initUndistortRectifyMap(rk,rightDis,r2,p2,imageSize,cv.CV_32FC1)

    def rectify(self,image,c="L"):
        x=406
        y=248
        h=607
        w=1103
        if c == "L":
            image=cv.remap(image,self.leftMap1,self.leftMap2,cv.INTER_NEAREST)
        else:
            image=cv.remap(image,self.rightMap1,self.rightMap2,cv.INTER_NEAREST)
        finalimage = image[y:y+h, x:x+w]
        return finalimage

    def triangulate(self,leftimage,rightimage,mode="SGBM",lpoints=0,rpoints=0):
        # flength=1403.11
        # baseline=0.11924492
        if mode == "SGBM":
            stereo = cv.StereoSGBM_create(numDisparities=16*6, blockSize=15,P1=4000,P2=15000,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
            disparitymap = stereo.compute(leftimage,rightimage)
            return disparitymap 
            # display(disparitymap/255,"Disparity Map")
            disparitymap[disparitymap == 0.0] = 0.1
            disparitymap[disparitymap == -1.0] = 0.1
        
            depth = np.ones(disparitymap.shape)
            depth= self.flength * self.baseline / (disparitymap+0.00000001)
        else:
            if len(lpoints) == len(rpoints):
                depth=[]
                for i in range(len(lpoints)):
                    depth.append((flength*baseline)/((lpoints[i][0]-rpoints[i][0])+0.00000001))

            else:
                print("Different number of features!")
        return depth


if __name__ == "__main__":

    cammodel=camera()
    # exit(0)
    count=0
    extansion='*.jpg'
    dataDir="/Users/alpha/Downloads/Data_to_be_shared/Indian_Road_Stereo_lite/left/"+extansion
    filepaths=glob.glob(dataDir)
    filepath=filepaths.sort()
    for filepath in filepaths:
        leftimage=cv.imread(filepath,0)
        filepath=filepath.replace('left','right')
        rightimage=cv.imread(filepath,0)
        count+=1
        leftimage=cammodel.rectify(leftimage,"L")
        rightimage=cammodel.rectify(rightimage,"R")


        pointsL,pointsR=detectFeature(leftimage,rightimage)
        # point3D=triangulate(leftimage,rightimage,"NO",pointsL,pointsR)
        point3D=cammodel.triangulate(leftimage,rightimage)
        display(point3D)
        # print(point3D)
        # testing
        xdata=[]
        ydata=[]
        zdata=[]
        ax = plt.axes(projection='3d')
        for i in range(len(point3D)):
            xdata.append(pointsL[i][0])
            ydata.append(pointsL[i][1])
            zdata.append(point3D[i])
        #3D Visualize    
        # to3d.convert_to_3d(leftimage,point3D)
        # exit(0)
        # ax.scatter3D(xdata, ydata, zdata, cmap="gray",marker='.',linewidths=1)
        # plt.xlim(-1920, 1920)
        # plt.ylim(-1080, 1080)
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()

    print('Processed images:',count)