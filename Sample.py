
# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('/Users/a_jing/Documents/Courses/CSE 252C/Project/code/Ours/Rect.py')
import Rect
sys.path.append('/Users/a_jing/Documents/Courses/CSE 252C/Project/code/Ours/ImageRep.py')
import ImageRep

class Sample:
    def __init__(self, image, roi):
        self.m_image = ImageRep(image)
        self.m_roi = Rect(roi)
    def GetImage(self):
        return self.m_image
    def GetROI(self):
        return self.m_roi
    
class MultiSample:
    def __init__(self, image, rects):
        self.m_image = ImageRep(image)
        self.m_rects = rects
    def GetImage(self):
        return self.m_image
    def GetRects(self):
        return self.m_rects
    def GetSample(i):
        return Sample(self.m_image, self.m_rects[i])
    

