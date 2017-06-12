import sys
import cv2 as cv
import numpy as np
sys.path.append('./Features.py')
from Features import Features
sys.path.append('./Config.py')
from Config import Config
sys.path.append('./Sample.py')
from Sample import Sample
sys.path.append('./Rect.py')
from Rect import Rect


class MultiFeatures(Features):
	def __init__(self, features):
		Features.__init__(self)
		self.m_features = features
		d = 0
		for i in range(len(features)):
			d += features[i].GetCount()
			self.SetCount(d)

	def UpdateFeature(self, sam):
		start = 0
		for i in range(len(self.m_features)):
			n = self.m_features[i].GetCount()
			self.m_featVec[start:start+n] = self.m_features[i].Eval(sam)
			start += n