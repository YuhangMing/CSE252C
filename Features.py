import sys
import cv2 as cv
import numpy as np
sys.path.append('./Sample.py')
from Sample import Sample

class Features:
    def __init__(self):
        self.m_featureCount = 0
        self.m_featVec = []

    def SetCount(self, c):
        self.m_featureCount = c
        self.m_featVec = [0]*c

    def GetCount(self):
        return self.m_featureCount

    def UpdateFeature(self, sam):
        pass

    # s - multisample
    def Eval(self, s, featVecs=None):
        if featVecs is None:
            self.UpdateFeature(s)
            return self.m_featVec
        else:
            # ???
            featVecs = []
            for i in range(len(s.GetRects())):
                print(self.Eval(s.GetSample(i)))
                featVecs.append(self.Eval(s.GetSample(i)))

