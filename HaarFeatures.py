import sys
import numpy as np

sys.path.append('./Config.py')
from Config import Config
sys.path.append('./HaarFeature.py')
from HaarFeature import HaarFeature
sys.path.append('./Features.py')
from Features import Features


class HaarFeatures(Features):
	
	kSystematicFeatureCount = 192

	## self.m_features -> self.features 
	## self.featureCount -> self.featureCount
	# m_featVec -> featList
	def __init__(self, conf):
		self.features = []	############# delete
		# self.featList = []		############# delete
		# Skip Features Class in original C++ Code !!!!!!!!!!
		# SetCount(kSystematicFeatureCount)
		self.featureCount = 192
		# # m_featureCount = kSystematicFeatureCount
		self.GenerateSystematic()

	def add_feature(self, new_feature):
		self.features.append(new_feature)
	# def add_featVec(self, new_featVec):
	# 	self.featList.append(new_featVec)

	# def GetCount(self):
	# 	return self.featureCount

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

	def UpdateFeature(self, sample):
		self.featList = []
		for i in range(self.featureCount):
			new_featList = self.features[i].Eval(sample)
			self.featList.append(new_featList)
			# self.add_featVec(new_featVec)
		# print('length of m_featVec: '+str(len(self.featList)))
		# return self.featList


	# def EvalOne(self, s):
	# 	return self.UpdateFeatureVector(s)


	# def Eval(self, s, featVec):
	# 	print("haarfeatures")
	# 	# print('# of rects: ' + str(len(s.GetRects())))
	# 	for j in range(len(s.GetRects())):
	# 		# featVec[i, :] = self.UpdateFeatureVector(s.GetSample(i))
	# 		# featVec[i, :] = self.EvalOne(s.GetSample(i))
	# 		self.featList = []
	# 		for i in range(self.featureCount):
	# 			new_featVec = self.features[i].Eval(s.GetSample(j))
	# 			self.featList.append(new_featVec)
	# 		featVec[j, :] = self.featList
			# print("in haarfeatures.Eval, j = : ", j)
			# print(featVec[j, :])
			# print("")






