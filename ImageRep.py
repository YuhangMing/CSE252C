# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect

kNumBins = 16;
class ImageRep:
    def __init__(self, image, computeIntegral=True, computeIntegralHist=False, colour=False):
        # print("ImageRep.init")
        # print(computeIntegral, computeIntegralHist, colour)
        self.channels = 3 if colour else 1
        self.rect = Rect(0, 0, image.shape[1], image.shape[0])
        # print(self.rect.XMin())
        self.images, self.integralImages, self.integralHistImages = [],[],[]

        for i in range(self.channels):
            # self.images.append(cv.createMat(image.shape[0], image.shape[1], cv.CV_8UC1))
            self.images.append(np.zeros((image.shape[0], image.shape[1]), np.uint8))
            if computeIntegral:
                # self.integralImages.append(cv.createMat(image.shape[0]+1, image.shape[1]+1, cv.CV_32SC1))
                self.integralImages.append(np.zeros((image.shape[0]+1, image.shape[1]+1), np.float32))
            if computeIntegralHist:
                for j in range(kNumBins):
                    # self.integralHistImages.append(cv.createMat(image.shape[0]+1, image.shape[1]+1, cv.CV_32SC1))
                    self.integralHistImages.append(np.zeros((image.shape[0]+1, image.shape[1]+1), np.float32))

        if len(image.shape) == 2:
            channels = 1
        else:
            channels = 3
        if colour:
            # assert(image.channels() == 3)
            assert(channels == 3)
            b, g, r = cv.split(image)
            self.images = [b, g, r]
        else:
            #assert(image.channels() == 3 or image.channels() == 1)
            assert(channels == 3 or channels == 1)
            if channels == 3:
                self.images[0] = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            elif channels == 1:
                np.copyto(self.images[0], image)

        if computeIntegral:
            for i in range(self.channels):
                self.images[i] = cv.equalizeHist(self.images[i])
                self.integralImages[i] = cv.integral(self.images[i])

        if computeIntegralHist:
            # tmp = cv.createMat(image.shape[0], image.shape[1], cv.CV_8UC1)
            # tmp[:] = 0
            tmp = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            for j in range(kNumBins):
                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        sbin = int(float(self.images[0][y][x])/256.0*kNumBins)
                        tmp[y][x] = 1 if sbin == j else 0
                self.integralHistImages[j] = cv.integral(tmp)


    def Sum(self, rRect, channel=0):
        #rRect.printStr()
        #print self.images[0].shape
        assert(rRect.getX()>=0 and rRect.getY()>=0 and                rRect.getXMax()<=self.images[0].shape[1] and rRect.getYMax()<=self.images[0].shape[0])
        return self.integralImages[channel][rRect.getY()][rRect.getX()] +                 self.integralImages[channel][rRect.getYMax()][rRect.getXMax()] -                 self.integralImages[channel][rRect.getYMax()][rRect.getX()] -                 self.integralImages[channel][rRect.getY()][rRect.getXMax()]

    def Hist(self, rRect):
        assert(rRect.getX()>=0 and rRect.getY()>=0 and                 rRect.getXMax()<=self.images[0].shape[1] and rRect.getYMax()<=self.images[0].shape[0])
        norm = rRect.getArea();
        h = [0]*kNumBins
        for i in range(kNumBins):
            total = self.integralHistImages[i][int(rRect.getY())][int(rRect.getX())] +                     self.integralHistImages[i][int(rRect.getYMax())][int(rRect.getXMax())] -                     self.integralHistImages[i][int(rRect.getYMax())][int(rRect.getX())] -                     self.integralHistImages[i][int(rRect.getY())][int(rRect.getXMax())]
            h[i] = float(total)/norm;
        return h

    def GetImage(self, channel=0):
        return self.images[channel]

    def GetRect(self):
        return self.rect;
