
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
    def __init__(self, fRect, iType):
        assert(iType<6)
        # self.fRect = Rect(fRect)
        self.fRect = Rect()
        self.fRect.initFromList(fRect)
        self.m_weights, self.m_rects = [], []
        if iType == 0:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width(), self.fRect.Height()/2))
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin()+self.fRect.Height()/2, self.fRect.Width(), self.fRect.Height()/2))
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType == 1:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width()/2, self.fRect.Height()))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()/2, self.fRect.YMin(), self.fRect.Width()/2, self.fRect.Height()))
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType == 2:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width()/3, self.fRect.Height()))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()/3, self.fRect.YMin(), self.fRect.Width()/3, self.fRect.Height()))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()*2/3, self.fRect.YMin(), self.fRect.Width()/3, self.fRect.Height()))
            self.m_weights.append(1.0)
            self.m_weights.append(-2.0)
            self.m_weights.append(1.0)
            self.m_factor = 255.0*2.0/3.0
        elif iType == 3:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width(), self.fRect.Height()/3))
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin()+self.fRect.Height()/3, self.fRect.Width(), self.fRect.Height()/3))
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin()+self.fRect.Height()*2/3, self.fRect.Width(), self.fRect.Height()/3))
            self.m_weights.append(1.0)
            self.m_weights.append(-2.0)
            self.m_weights.append(1.0)
            self.m_factor = 255.0*2.0/3.0
        elif iType == 4:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width()/2, self.fRect.Height()/2))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()/2, self.fRect.YMin()+self.fRect.Height()/2, self.fRect.Width()/2, self.fRect.Height()/2))
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin()+self.fRect.Height()/2, self.fRect.Width()/2, self.fRect.Height()/2))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()/2, self.fRect.YMin(), self.fRect.Width()/2, self.fRect.Height()/2))
            self.m_weights.append(1.0)
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType == 5:
            self.m_rects.append(Rect(self.fRect.XMin(), self.fRect.YMin(), self.fRect.Width(), self.fRect.Height()))
            self.m_rects.append(Rect(self.fRect.XMin()+self.fRect.Width()/4, self.fRect.YMin()+self.fRect.Height()/4, self.fRect.Width()/2, self.fRect.Height()/2))
            self.m_weights.append(1.0)
            self.m_weights.append(-4.0)
            self.m_factor = 255.0*3.0/4.0
            
    
    def Eval(self, sam):
        image = sam.GetImage()
        roi = sam.GetROI()
        value = 0.0
        for i in range(len(self.m_rects)):
            fRect = self.m_rects[i]
            samRect = Rect(int(roi.XMin()+fRect.XMin()*roi.Width()+0.5), int(roi.YMin()+fRect.YMin()*roi.Height()+0.5), int(fRect.Width()*roi.Width()), int(fRect.Height()*roi.Height()))
            value += self.m_weights[i]*image.Sum(samRect)
            # print(" i = ", i)
            # print(roi.XMin(), roi.YMin(), roi.Width(), roi.Height())
            # print(fRect.XMin(), fRect.YMin(), fRect.Width(), fRect.Height())
            # print(int(roi.XMin()+fRect.XMin()*roi.Width()+0.5), int(roi.YMin()+fRect.YMin()*roi.Height()+0.5), int(fRect.Width()*roi.Width()), int(fRect.Height()*roi.Height()))

        # print(value)
        # print("in haarfeature.eval: ")
        # print(value / (self.m_factor*roi.Area()*self.fRect.Area()))
        # print( "")
        
        return value / (self.m_factor*roi.Area()*self.fRect.Area())


