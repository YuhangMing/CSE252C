import sys
import cv2 as cv
sys.path.append('./Sample.py')
from Sample import Sample

class Feature:
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

    def Eval(self, multisam, feats=None):
    	if feats is None:
    		self.UpdateFeature(multisam)
    		return self.m_featVec
    	else:
            # ???
    		for i in (len(multisam.GetRects())):
    			self.m_featVec.append(Eval(multisam.GetSample(i)))
