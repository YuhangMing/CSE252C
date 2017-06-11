import sys
import cv2 as cv
sys.path.append('./Sample.py')
from Sample import Sample

class Feature:
    def __init__(self):
    	self.m_featureCount = 0

    def SetCount(self, c):
    	self.m_featureCount = c
    	self.m_feat = []

    def GetCount(self):
    	return self.m_featureCount

    def UpdateFeature(self, sam):
    	pass

    def Eval(self, multisam, feats=None):
    	if feats is None:
    		self.UpdateFeature(multisam)
    		return self.m_feat
    	else:
    		for i in range(len(multisam.GetRects())):
    			self.m_feat.append(Eval(multisam.GetSample(i)))
