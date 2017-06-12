import sys
import cv2 as cv
import numpy as np
sys.path.append('./Sample.py')
from Sample import Sample


class Features:
    # self.m_featureCount -> self.featureCount
    # self.m_featVec -> self.featList
    def __init__(self):
        self.featureCount = 0
        self.featList = []

    def SetCount(self, c):
        self.featureCount = c
        self.featList = [0]*c

    def GetCount(self):
        return self.featureCount

    def UpdateFeature(self, sam):
        pass

    # s - multisample
    def Eval(self, s, featLists=None):
        if featLists is None:
            self.UpdateFeature(s)
            return self.featList
        else:
            for i in xrange(len(s.GetRects())):
                featLists.append(self.Eval(s.GetSample(i)))