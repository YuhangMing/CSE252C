import sys
import numpy as np

sys.path.append('./Config.py')
from Config import Config
sys.path.append('./HaarFeature.py')
from HaarFeature import HaarFeature


class HaarFeatures:
	
	kSystematicFeatureCount = 192

	def __init__(self, conf):
		self.m_features = []
		self.m_featVec = []
		# Skip Features Class in original C++ Code !!!!!!!!!!
		# SetCount(kSystematicFeatureCount)
		self.m_featureCount = 192
		# # m_featureCount = kSystematicFeatureCount
		self.GenerateSystematic()

	def add_feature(self, new_feature):
		self.m_features.append(new_feature)
	# def add_featVec(self, new_featVec):
	# 	self.m_featVec.append(new_featVec)

	def GetCount(self):
		return self.m_featureCount

	def GenerateSystematic(self):
		x = [0.2, 0.4, 0.6, 0.8]
		y = [0.2, 0.4, 0.6, 0.8]
		s = [0.2, 0.4]
		for iy in range(4):
			for ix in range(4):
				for iscale in range(2):
					r = [x[ix]-s[iscale]/2, y[iy]-s[iscale]/2, s[iscale], s[iscale]]
					# r = Rect()
					# r.iniFromRect(r_value)
					for itype in range(6):
						new_feature = HaarFeature(r, itype)
						self.add_feature(new_feature)

	# def UpdateFeatureVector(self, sample):
	# 	self.m_featVec = []
	# 	for i in range(self.m_featureCount):
	# 		new_featVec = self.m_features[i].Eval(sample)
	# 		self.m_featVec.append(new_featVec)
	# 		# self.add_featVec(new_featVec)
	# 	# print('length of m_featVec: '+str(len(self.m_featVec)))
	# 	return self.m_featVec


	# def EvalOne(self, s):
	# 	return self.UpdateFeatureVector(s)


	def Eval(self, s, featVec):
		# print('# of rects: ' + str(len(s.GetRects())))
		for j in range(len(s.GetRects())):
			# featVec[i, :] = self.UpdateFeatureVector(s.GetSample(i))
			# featVec[i, :] = self.EvalOne(s.GetSample(i))
			self.m_featVec = []
			for i in range(self.m_featureCount):
				new_featVec = self.m_features[i].Eval(s.GetSample(j))
				self.m_featVec.append(new_featVec)
			featVec[j, :] = self.m_featVec
			# print("in haarfeatures.Eval, j = : ", j)
			# print(featVec[j, :])
			# print("")






