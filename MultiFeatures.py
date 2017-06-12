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
	## self.m_features -> self.features
	## self.m_featVec -> self.featList
	def __init__(self, features):
		Features.__init__(self)
		self.features = features
		d = 0
		for i in range(len(features)):
			d += features[i].GetCount()
			self.SetCount(d)

	def UpdateFeature(self, sam):
		start = 0
		for i in range(len(self.features)):
			n = self.features[i].GetCount()
			self.featList[start:start+n] = self.features[i].Eval(sam)
			start += n