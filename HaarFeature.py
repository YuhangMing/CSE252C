
# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./ImageRep.py')
from ImageRep import ImageRep
sys.path.append('./Sample.py')
from Sample import Sample

class HaarFeature:
    ## self.m_weights -> self.weights
    ## self.m_rects -> self.rects
    ## self.m_factor -> self.factor
    def __init__(self, fRect, iType):
        assert(iType<6)
        # self.fRect = Rect(fRect)
        self.fRect = Rect()
        self.fRect.initFromList(fRect)
        self.weights, self.rects = [], []
        if iType == 0:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth(), self.fRect.getHeight()/2.0))
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY()+self.fRect.getHeight()/2.0, self.fRect.getWidth(), self.fRect.getHeight()/2.0))
            self.weights.append(1.0)
            self.weights.append(-1.0)
            self.factor = 255.0/2.0
        elif iType == 1:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth()/2.0, self.fRect.getHeight()))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()/2.0, self.fRect.getY(), self.fRect.getWidth()/2.0, self.fRect.getHeight()))
            self.weights.append(1.0)
            self.weights.append(-1.0)
            self.factor = 255.0/2.0
        elif iType == 2:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth()/3.0, self.fRect.getHeight()))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()/3.0, self.fRect.getY(), self.fRect.getWidth()/3.0, self.fRect.getHeight()))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()*2.0/3.0, self.fRect.getY(), self.fRect.getWidth()/3.0, self.fRect.getHeight()))
            self.weights.append(1.0)
            self.weights.append(-2.0)
            self.weights.append(1.0)
            self.factor = 255.0*2.0/3.0
        elif iType == 3:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth(), self.fRect.getHeight()/3.0))
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY()+self.fRect.getHeight()/3.0, self.fRect.getWidth(), self.fRect.getHeight()/3.0))
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY()+self.fRect.getHeight()*2.0/3.0, self.fRect.getWidth(), self.fRect.getHeight()/3.0))
            self.weights.append(1.0)
            self.weights.append(-2.0)
            self.weights.append(1.0)
            self.factor = 255.0*2.0/3.0
        elif iType == 4:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth()/2.0, self.fRect.getHeight()/2.0))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()/2.0, self.fRect.getY()+self.fRect.getHeight()/2.0, self.fRect.getWidth()/2.0, self.fRect.getHeight()/2.0))
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY()+self.fRect.getHeight()/2.0, self.fRect.getWidth()/2.0, self.fRect.getHeight()/2.0))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()/2.0, self.fRect.getY(), self.fRect.getWidth()/2.0, self.fRect.getHeight()/2.0))
            self.weights.append(1.0)
            self.weights.append(1.0)
            self.weights.append(-1.0)
            self.weights.append(-1.0)
            self.factor = 255.0/2.0
        elif iType == 5:
            self.rects.append(Rect(self.fRect.getX(), self.fRect.getY(), self.fRect.getWidth(), self.fRect.getHeight()))
            self.rects.append(Rect(self.fRect.getX()+self.fRect.getWidth()/4.0, self.fRect.getY()+self.fRect.getHeight()/4.0, self.fRect.getWidth()/2.0, self.fRect.getHeight()/2.0))
            self.weights.append(1.0)
            self.weights.append(-4.0)
            self.factor = 255.0*3.0/4.0
            
    
    def Eval(self, sam):
        image = sam.GetImage()
        roi = sam.GetROI()
	'''
        value = 0.0
	#print(len(self.rects))
        for i in range(len(self.rects)):
            fRect = self.rects[i]
            samRect = Rect(int(roi.getX()+fRect.getX()*roi.getWidth()+0.5), int(roi.getY()+fRect.getY()*roi.getHeight()+0.5), int(fRect.getWidth()*roi.getWidth()), int(fRect.getHeight()*roi.getHeight()))
            value += self.weights[i]*image.Sum(samRect)
	'''
    	samRects = list(map(lambda r: Rect(int(roi.getX()+r.getX()*roi.getWidth()+0.5), int(roi.getY()+r.getY()*roi.getHeight()+0.5), int(r.getWidth()*roi.getWidth()), int(r.getHeight()*roi.getHeight())), self.rects))

    	srs = list(map(lambda sr: image.Sum(sr), samRects))
    	value = np.dot(np.array(self.weights), np.array(srs))
        
        return value / (self.factor*roi.getArea()*self.fRect.getArea())
