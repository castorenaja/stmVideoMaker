#!/usr/bin/python

import os.path
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
from pylab import *
from PIL import Image
import pylab
import math
import matplotlib.gridspec as gridspec

def videoReader(inVideo, inSTM, fps):

    myImage = cv2.imread(inSTM,0) # 1:Color, 0:Grayscale, -1:Unchaged
    myH,myW = myImage.shape
 
    directory = './tifs/'

    dir = os.path.dirname(directory)
    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        cap = cv2.VideoCapture(inVideo)
        frW  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except:
        print 'Error Loading Video File'

    dimDiff = float(frW-myH)/2.

    if dimDiff.is_integer():
        if dimDiff>0:
            onesExt = np.ones(shape=(int(dimDiff),myW))
            onesExt = onesExt*255
            HExt = onesExt.astype('uint8')
            extImage = np.append(HExt,myImage,0)
            extImage = np.append(extImage,HExt,0)
            print extImage.shape
        else:
            extImage = myImage
    else:
        onesExtTop = np.ones(shape=(int(dimDiff)+1,myW))
        onesExtTop = onesExtTop*255
        THExt = onesExtTop.astype('uint8')
        onesExtBott = np.ones(shape=(int(dimDiff),myW))
        onesExtBott = onesExtBott*255
        BHExt = onesExtBott.astype('uint8')
        extImage = np.append(THExt,myImage,0)
        extImage = np.append(extImage,BHExt,0)

    spacer = np.ones(shape=(frW,5))
    spacer = spacer*255
    spacerInt = spacer.astype('uint8')
            

    cap = cv2.VideoCapture(inVideo)
    cv2.namedWindow('videoFrames', cv2.CV_WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    nframe = 0
    oneFrame = 1/fps

    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret==True):
            nframe = nframe + 1

            if nframe in range(0,10):
                strNumFrame = '000'+str(nframe)
            if nframe in range(10,100):
                strNumFrame = '00'+str(nframe)
            if nframe in range(100,1000):
                strNumFrame = '0'+str(nframe)
            if nframe in range(1000,10000):
                strNumFrame = str(nframe)
                
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting to GrayScale
            grayFrame = np.rot90(grayFrame)
            tempImage = np.copy(extImage)
            tempImage[:,nframe:myW-1] = 255
            updatedFrame = np.append(grayFrame,spacerInt,1)
            concaImg = np.concatenate((updatedFrame,tempImage),axis=1)

            # Inserting TimeStamp

            concaH,concaW = concaImg.shape

            cv2.rectangle(concaImg,(concaW-150,concaH-50),(concaW,concaH),(0,0,0),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            timer = oneFrame*(nframe-1)
            cv2.putText(concaImg,'{:.2f}'.format(timer)+' s',(concaW-150,concaH-16), font, 1,(255,255,255),2)
            
            cv2.imshow('videoFrames',concaImg)                       
            uint8Img = concaImg.astype('uint8')
            imToWrite = Image.fromarray(uint8Img)
            imToWrite.save(directory+inSTM[:-4]+'_'+strNumFrame+'.tif',compression = 'None')
            del tempImage, concaImg, uint8Img, imToWrite
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
            cv2.destroyAllWindows
            cv2.waitKey(1)

inVideo = sys.argv[1]
inSTM = sys.argv[2]
fps = float(sys.argv[3])

videoReader(inVideo, inSTM, fps)
        



