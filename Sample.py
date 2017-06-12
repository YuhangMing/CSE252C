
# coding: utf-8

# In[1]:

import sys
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./ImageRep.py')
from ImageRep import ImageRep


class Sample:
    def __init__(self, image, roi):
        self.image = image
        self.roi = roi
    def GetImage(self):
        return self.image
    def GetROI(self):
        return self.roi
    def GetSelf(self):
        return self
    
class MultiSample:
    def __init__(self, image, rects):
        self.image = image
        self.rects = rects
    def GetSelf(self):
        return self
    def GetImage(self):
        return self.image
    def GetRects(self):
        return self.rects
    def GetSample(self, i):
        return Sample(self.image, self.rects[i])
    

