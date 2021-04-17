import cv2 as cv
import os
import numpy as np
from keras.utils import np_utils
import pickle as pkl

def dump_into_pkl(data,name):
    outfile = open(name,'wb')
    pkl.dump(data, outfile)
    outfile.close()
    print(name + " dumped")

class Preprocessing :
    def __init__(self,path):
        self.pth=path
    def Scan_folder(self):
        self.classes=os.listdir(self.pth)
        self.labels=[i for i in range(len(self.classes))]
        self.label_dict=dict(zip(self.classes,self.labels))
        print(self.labels)
        self.imgsize=100
        
    def Resize(self):
        X=[]
        y=[]
        for clas in self.classes:
            self.folderpath=os.path.join(self.pth,clas)
            self.imgnames=os.listdir(self.folderpath)
            #print(self.imgnames)
            for img_name in self.imgnames:
                path=os.path.join(self.folderpath,img_name)
                img=cv.imread(path)
                try:
                    grayimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)           
                    resized=cv.resize(grayimg,(self.imgsize,self.imgsize))
                    X.append(resized)
                    y.append(self.label_dict[clas])
                except Exception as e:
                    print('Exception:',e)
        X=np.array(X)/255.0
        X=np.reshape(X,(X.shape[0],self.imgsize,self.imgsize,1))
        y=np.array(y)
        ynew=np_utils.to_categorical(y)
        dump_into_pkl(X,"ResizedImage.pkl")
        dump_into_pkl(ynew,"labels.pkl")
        dump_into_pkl(y,"label.pkl")
   