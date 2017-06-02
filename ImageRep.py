
# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect

kNumBins = 16;
class ImageRep:
    def __init__(self, image, computeIntegral, computeIntegralHists, colour):
        self.m_channels = 3 if colour else 1
        self.m_rect = Rect(0, 0, image.cols, image.rows)
        self.m_images, self.m_integralImages, self.m_integralHistImages = [],[],[]
        
        for i in range(self.m_channels):
            self.m_images.append(cv.createMat(image.rows, image.cols, cv.CV_8UC1))
            if computeIntegral:
                self.m_integralImages.append(cv.createMat(image.rows+1, image.cols+1, cv.CV_32SC1))
            if computeIntegralHist:
                for j in range(kNumBins):
                    self.m_integralHistImages.append(cv.createMat(image.rows+1, image.cols+1, cv.CV_32SC1))
        
        if colour:
            assert(image.channels() == 3)
            b, g, r = cv.split(image)
            self.m_images = [b, g, r]
        else:
            assert(image.channels() == 3 or image.channels() == 1)
            if image.channels() == 3:
                self.m_images[0] = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            elif image.channels() == 1:
                np.copyto(self.m_images[0], image)
        
        if computeIntegral:
            for i in range(self.m_channels):
                self.m_images[i] = cv.equalizeHist(self.m_images[i])
                self.m_integralImages[i] = cv.integral(self.m_images[i])
        
        if computeIntegralHist:
            tmp = cv.createMat(image.rows, image.cols, cv.CV_8UC1)
            tmp[:] = 0
            for j in range(kNumBins):
                for y in range(image.rows):
                    for x in range(image.cols):
                        sbin = int(float(self.m_images[0][y][x])/256.0*kNumBins)
                        tmp[y][x] = 1 if sbin == j else 0
                self.m_integralImages[j] = cv.integral(tmp)
   
    
    def Sum(self, rRect, channel=0):
        assert(rRect.XMin()>=0 and rRect.YMin()>=0 and                rRect.XMax()<=self.m_images[0].cols and rRect.YMax()<=self.m_images[0].rows)
        return self.m_integralImages[channel][rRect.YMin()][rRect.XMin()] +                 self.m_integralImages[channel][rRect.YMax()][rRect.XMax()] -                 self.m_integralImages[channel][rRect.YMax()][rRect.XMin()] -                 self.m_integralImages[channel][rRect.YMin()][rRect.XMax()]
    
    def Hist(self, rRect, h):
        assert(rRect.XMin()>=0 and rRect.YMin()>=0 and                 rRect.XMax()<=self.m_images[0].cols and rRect.YMax()<=slef.m_images[0].rows)
        norm = rRect.Area();
        for i in range(kNumBins):
            total = self.m_integralHistImages[i][rRect.YMin()][rRect.XMin()] +                     self.m_integralHistImages[i][rRect.YMax()][rRect.XMax()] -                     self.m_integralHistImages[i][rRect.YMax()][rRect.XMin()] -                     self.m_integralHistImages[i][rRect.YMin()][rRect.XMax()]
            h[i] = float(total)/norm;
    
    def GetImage(self, channel=0):
        return self.m_images[channel]
    def GetRect(self):
        return self.m_rect;

