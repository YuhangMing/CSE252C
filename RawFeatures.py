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

kPatchSize = 16

class RawFeatures(Features):
	def __init__(self, conf):
		Features.__init__(self)
		self.patchImage = np.zeros((kPatchSize, kPatchSize), np.uint8)
		self.SetCount(kPatchSize**2)

	def UpdateFeature(self, sam):
		rect = sam.GetROI()
		original = sam.GetImage().GetImage(0)[int(rect.XMin()):int(rect.XMax())+1, int(rect.YMin()):int(rect.YMax())+1]
		self.patchImage = cv.resize(original,(self.patchImage.shape[1],self.patchImage.shape[0] ))
		tmp = self.patchImage[:]
		np.reshape(tmp, (1, kPatchSize**2))
		# print tmp.shape
		tmp = tmp/255.0
		# np.divide(tmp, 255.0, out=tmp)
		self.featList = tmp.tolist()
# 		ind = 0
# 		for i in range(kPatchSize):
# 			for j in range(kPatchSize):
# 				pixel = self.patchImage[i][j]
# 				self.featList[ind] = float(pixel)/255
# 				ind += 1
