import cv2
import glob
import numpy as np

class Camera:
    def __init__(self,path,folder,extension):
        self.path=path
        self.img=None
        self.frm=extension
        self.items=0
        self.count=-1
        self.loaded=False
        self.getfolder=folder

    def scanfolder(self):
        if not self.loaded:
            fl_nm=self.path+'/*.'+self.frm
            self.files=glob.glob(fl_nm)
            self.items=len(self.files)
            self.loaded=True

        if self.items == 0:
            print('Wrong input !')
            return 'END'

        elif self.count+1 < self.items:
            self.files.sort()
            self.count+=1
            return self.files[self.count]
        else:
            return 'END'

    def get(self):
        if self.getfolder:
            file_p=self.scanfolder()
        else:
            file_p=self.path

        if file_p == 'END':
            return None,False

        image =cv2.imread(file_p)
        return image,True

if __name__ == "__main__":
    # Tested
    # Folder reading -[x]
    # Error handeling -[x]
    frame=Camera('/Users/alpha/Downloads/360_degree_view/CAM_BACK',True,'jpg')
    nxt=True
    while(nxt):
        img,nxt=frame.get()
        if nxt:
            cv2.imshow('View',img)
        if cv2.waitKey(30)==27:
            cv2.destroyAllWindows()
            break