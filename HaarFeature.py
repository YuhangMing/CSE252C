
# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('/Users/a_jing/Documents/Courses/CSE 252C/Project/code/Ours/Rect.py')
import Rect
sys.path.append('/Users/a_jing/Documents/Courses/CSE 252C/Project/code/Ours/ImageRep.py')
import ImageRep
sys.path.append('/Users/a_jing/Documents/Courses/CSE 252C/Project/code/Ours/Sample.py')
import Sample

class HaarFeature:
    def __init__(self, fRect, iType):
        assert(iType<6)
        self.fRect = Rect(fRect)
        self.m_weights, self.m_rects = [], []
        if iType is 0:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), fRect.Width(), int(fRect.Height()/2)))
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin()+int(fRect.Height()/2), fRect.Width(), int(fRect.Height()/2)))
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType is 1:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), int(fRect.Width()/2), fRect.Height()))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()/2), fRect.YMin(), int(fRect.Width()/2), fRect.Height()))
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType is 2:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), int(fRect.Width()/3), fRect.Height()))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()/3), fRect.YMin(), int(fRect.Width()/3), fRect.Height()))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()*2/3), fRect.YMin(), int(fRect.Width()/3), fRect.Height()))
            self.m_weights.append(1.0)
            self.m_weights.append(-2.0)
            self.m_weights.append(1.0)
            self.m_factor = 255.0*2.0/3.0
        elif iType is 3:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), fRect.Width(), int(fRect.Height()/3)))
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin()+int(fRect.Height()/3), fRect.Width(), int(fRect.Height()/3)))
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin()+int(fRect.Height()*2/3), fRect.Width(), int(fRect.Height()/3)))
            self.m_weights.append(1.0)
            self.m_weights.append(-2.0)
            self.m_weights.append(1.0)
            self.m_factor = 255.0*2.0/3.0
        elif iType is 4:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), int(fRect.Width()/2), int(fRect.Height()/2)))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()/2), fRect.YMin()+int(fRect.Height()/2), int(fRect.Width()/2), int(fRect.Height()/2)))
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin()+int(fRect.Height()/2), int(fRect.Width()/2), int(fRect.Height()/2)))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()/2), fRect.YMin(), int(fRect.Width()/2), int(fRect.Height()/2)))
            self.m_weights.append(1.0)
            self.m_weights.append(1.0)
            self.m_weights.append(-1.0)
            self.m_weights.append(-1.0)
            self.m_factor = 255.0/2.0
        elif iType is 5:
            self.m_rects.append(Rect(fRect.XMin(), fRect.YMin(), fRect.Width(), fRect.Height()))
            self.m_rects.append(Rect(fRect.XMin()+int(fRect.Width()/4), fRect.YMin()+int(fRect.Height()/4), int(fRect.Width()/2), int(fRect.Height()/2)))
            self.m_weights.append(1.0)
            self.m_weights.append(-4.0)
            self.m_factor = 255.0*3.0/4.0
            
    
    def Eval(self, sam):
        image = sam.GetImage()
        roi = sam.GetROI()
        value = 0.0
        for i in range(self.m_rects.size()):
            fRect = self.m_rects[i]
            samRect = Rect(int(roi.XMin()+r.XMin()*roi.Width()+0.5),                            int(roi.YMin()+r.YMin()*roi.Height()+0.5),                            int(r.Width()*roi.Width()),                            int(r.Height()*roi.Height()))
            value += self.m_weights[i]*image.Sum(samRect)
        return value / (self.m_factor*roi.Area()*self.fRect.Area())

