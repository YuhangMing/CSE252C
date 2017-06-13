import numpy as np
np.set_printoptions(threshold=np.nan)
import math
import cv2
import sys
import random

sys.path.append('./Config.py')
from Config import Config
sys.path.append('./Features.py')
from Features import Features
sys.path.append('./Sample.py')
from Sample import Sample
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./GraphUtils.py')
import GraphUtils


class SupportPattern:
	def __init__(self):
		self.x = []			# value of features
		self.yv = []		# floadRect
		self.images = []	# Mat
		self.y = 0			# int
		self.refCount = 0	# int, num of svs in this pattern

	def add_x(self, new_x):
		self.x.append(new_x)
	def add_yv(self, new_yv):
		self.yv.append(new_yv)
	def add_image(self, new_image):
		self.images.append(new_image)

class SupportVector:
	def __init__(self, x=0.0, y=0.0, b=0.0, g=0.0, image = []):
		self.x = x 			# supportPattern
		self.y = y 			# int, index of the rects
		self.b = b 			# double, beta
		self.g = g 			# double, value of the gradient(mini)
		self.image = image 	# Mat

kMaxSVs = 2000
kTileSize = 30

class LaRank:
	def __init__(self, conf, features, kernel):
		self.config_file = conf 											# Configuration class
		self.feature_file = features 										# Feature class
		self.kernel_type = kernel 											# Kernel class
		self.svm_C = float(conf.svmC) 										# slack variable in svm
		N = conf.svmBudgetSize + 2 if conf.svmBudgetSize > 0 else kMaxSVs	# max size of kernel matrix
		self.kernelMat = np.zeros((N, N), dtype=np.float32) 				# Kernal matrix
		self.debugImage = np.zeros((800, 600, 3), np.uint8) 				# sv images shown in learner window

		self.sps = []
		self.svs = []

	def add_sps(self, new_sp):
		self.sps.append(new_sp)
	def add_svs(self, new_sv):
		self.svs.append(new_sv)

	# calculate the Discreminant function F(x, y)
	def CalF(self, x, y):
		f = 0.0
		for i in range(len(self.svs)):
			sv = self.svs[i]
			f += sv.b * self.kernel_type.Eval(x, sv.x.x[sv.y]) # Eval function in kernel class
		return f

	def Eval(self, sample, results): 
		centre = Rect()
		centre.initFromRect(sample.GetRects()[0])
		fvs = []
		self.feature_file.Eval(sample, fvs)	# Eval function in Features, results in fvs variable
		results[:] = []
		for i in range(len(fvs)):
			# express y in coord fram of center sample
			y = Rect()
			y.initFromRect( sample.GetRects()[i] )
			y.translate(-centre.getX(), -centre.getY())	# functions in Rect
			results.append(self.CalF(fvs[i], y))

	# i, y+, y-
	def SMOStep(self, yp, yn):
		if (yp == yn):
			return 
		svp = self.svs[yp]
		svn = self.svs[yn]
		# svs must in the same sp
		assert (svp.x == svn.x)
		sp = svp.x

		# print("SMO: gpos: %d gneg:  " % (svp.g, svp.g))
		if ((svp.g - svn.g) < 1e-5):
			# print("SMO: skipping")
			pass
		else:
			k00 = self.kernelMat[yp, yp]
			k11 = self.kernelMat[yn, yn]
			k01 = self.kernelMat[yp, yn]
			lambda_u = (svp.g - svn.g) / (k00 + k11 - 2 * k01)
			lambda_max = max(0, min(lambda_u, self.svm_C * float(svp.y == sp.y) - svp.b))

			# update coefficients
			svp.b += lambda_max
			svn.b -= lambda_max

			# update gradient
			for i in range(len(self.svs)):
				svi = self.svs[i]
				k0 = self.kernelMat[i, yp]
				k1 = self.kernelMat[i, yn]
				svi.g -= lambda_max * (k0 - k1)
			# print("SMO: %d, %d -- %f, %f (%f)" % (yp, yn, svp.b, svn.b, l))

		# beta < 1e-8 is consider to be 0
		if (abs(svp.b) < 1e-8):
			self.RemoveSupportVector(yp)
			if (yn == len(self.svs)):
				# yn and yp will have been swapped during sv removal
				yn = yp
		if (abs(svn.b) < 1e-8):
			self.RemoveSupportVector(yn)

	# min_y g_i(y)
	def MinGradient(self, ind):
		sp = self.sps[ind]
		minGrad = sys.float_info.max
		for i in range(len(sp.yv)):
			grad = -(1 - sp.yv[i].overlap(sp.yv[sp.y])) - self.CalF(sp.x[i], sp.yv[i])
			if (grad < minGrad):
				minInd = i
				minGrad = grad
		return minInd, minGrad

	def ProcessNew(self, ind):
		# gradient is -F(x, y) since loss = 0
		# print("process new ind = ", ind)
		# y+ = yi, because adding new sv, loss function is 0
		yp = self.AddSupportVector(self.sps[ind], self.sps[ind].y, -self.CalF(self.sps[ind].x[ self.sps[ind].y ], self.sps[ind].yv[ self.sps[ind].y ] ))
		# AddSupportVector(support pattern, rect index, gradient):
		# y- = argmin_y g_i(y)
		minInd, minGrad = self.MinGradient(ind)
		yn = self.AddSupportVector(self.sps[ind], minInd, minGrad)

		self.SMOStep(yp, yn)

	def ProcessOld(self):
		# operates on exsit support pattern
		if (len(self.sps) == 0):
			return

		# randomly choose pattern to process
		ind = random.randrange(len(self.sps))

		# find existing sv with largest grad and nonzero beta
		yp = -1
		maxGrad = -sys.float_info.max
		for i in range(len(self.svs)):
			if (self.svs[i].x != self.sps[ind]):
				continue

			svi = self.svs[i]
			if (svi.g > maxGrad and svi.b < self.svm_C * float(svi.y == self.sps[ind].y)):
				yp = i
				maxGrad = svi.g

		assert (yp != -1)
		if (yp == -1):
			return 

		# find potentially new sv with smallest grad
		minInd, minGrad = self.MinGradient(ind)
		yn = -1
		for i in range(len(self.svs)):
			if (self.svs[i].x != self.sps[ind]):
				continue 

			if (self.svs[i].y == minInd):
				yn = i
				break

		# add new sv
		if (yn == -1):
			yn = self.AddSupportVector(self.sps[ind], minInd, minGrad)

		self.SMOStep(yp, yn)

	def Optimize(self):
		if (len(self.sps) == 0):
			return

		# choose pattern to optimize
		ind = random.randrange(len(self.sps))

		yp = -1
		yn = -1
		maxGrad = -sys.float_info.max
		minGrad = sys.float_info.max

		# print(len(self.svs))

		for i in range(len(self.svs)):
			if (self.svs[i].x != self.sps[ind]):	# search among the support patterns
				continue

			svi = self.svs[i]
			if (svi.g > maxGrad and svi.b < self.svm_C * float(svi.y == self.sps[ind].y)):
				yp = i
				maxGrad = svi.g

			if (svi.g < minGrad):
				yn = i
				minGrad = svi.g

		assert (yp != -1 and yn != -1)
		if (yp == -1 or yn == -1):
			# this should not happen
			print("!!!!!!!!!!!!!!!!!!!!!!!")
			return

		self.SMOStep(yp, yn)

	def BudgetMaintenance(self):
		if (self.config_file.svmBudgetSize > 0):
			# remove a support vector if the maximum number of sv is exceeded
			while (len(self.svs) > self.config_file.svmBudgetSize):
				# find negative sv with smallest effect on discriminant function if removed
				minVal = sys.float_info.max
				yn = -1
				yp = -1
				for i in range(len(self.svs)):
					if (self.svs[i].b < 0.0):
						# find corresponding postive sv
						j = -1
						for k in range(len(self.svs)):
							if (self.svs[k].b > 0.0 and self.svs[k].x == self.svs[i].x):
								j = k
								break
						
						val = (self.svs[i].b ** 2) * (self.kernelMat[i, i] + self.kernelMat[j, j] - 2.0 * self.kernelMat[i, j])
						if (val < minVal):
							minVal = val
							yn = i
							yp = j

				# adjust weight of positive sv to compensate for removal of negative
				self.svs[yp].b += self.svs[yn].b

				# remove negative sv
				self.RemoveSupportVector(yn)
				# yp and yn will have been swapped during sv removal
				if (yp == len(self.svs)):
					yp = yn
				# also remove positive sv
				if (self.svs[yp].b < 1e-8):
					self.RemoveSupportVector(yp)

				# update gradients
				for i in range(len(self.svs)):
					svi = self.svs[i]
					svi.g = -(1 - svi.x.yv[svi.y].overlap(svi.x.yv[svi.x.y])) - self.CalF(svi.x.x[svi.y], svi.x.yv[svi.y])

	# Update Discriminant Function #### CORE ####
	def Update(self, sample, y):
		# print('start updating')
		# create Support Pattern
		sp = SupportPattern()
		rects = sample.GetRects()	# GetRects function in Sample class, a list of rects, 4xn
		center = Rect()
		center.initFromRect(rects[y])
		for i in range(len(rects)):
			r = Rect()
			r.initFromRect(rects[i])
			# represent rectangle in the coordinate frame of the center rectangle
			r.translate(-center.getX(), -center.getY())
			sp.add_yv(r)
			if (not(self.config_file.quietMode) and self.config_file.debugMode):
				im = np.zeros((kTileSize, kTileSize), np.uint8)
				rect = Rect()
				rect.initFromRect(rects[i])

				roi = [rect.getX(), rect.getX()+rect.getWidth(), rect.getY(), rect.getY()+rect.getHeight()] #[xmin, xmax, ymin, ymax]
				cv2.resize(sample.GetImage().GetImage(0)[int(roi[2]):int(roi[3]), int(roi[0]):int(roi[1])], im.shape, im)
				sp.add_image(im)

		# evaluating feature for each sample
		sp.x = []
		self.feature_file.Eval(sample, sp.x)

		sp.y = y
		sp.refCount = 0
		self.add_sps(sp)

		self.ProcessNew( len(self.sps)-1 )
		self.BudgetMaintenance()

		for i in range(10):
			# Reprocess
			self.ProcessOld()
			self.BudgetMaintenance()
			for j in range(10):
				self.Optimize()

	def AddSupportVector(self, x, y, g):
		sv = SupportVector(x, y, 0.0, g) 
		# (support pattern, index of the rect, beta, gradient)

		ind = len(self.svs)
		self.add_svs(sv)
		x.refCount += 1
		# print("Adding SV: ", ind)
		# print('y = %d, g = %f' % (y, g))

		# update kernel matrix
		for i in range(ind):
			self.kernelMat[i, ind] = self.kernel_type.Eval(self.svs[i].x.x[ self.svs[i].y ], x.x[y])
			self.kernelMat[ind, i] = self.kernelMat[i, ind]
		
		self.kernelMat[ind, ind] = self.kernel_type.Eval( x.x[y] )
		return ind

	def SwapSupportVectors(self, ind1, ind2):
		# swap svs
		tmp = self.svs[ind1];
		self.svs[ind1] = self.svs[ind2];
		self.svs[ind2] = tmp;
		
		# swap row and col in kernel matrix
		row1 = self.kernelMat[ind1, :];
		self.kernelMat[ind1, :] = self.kernelMat[ind2, :];
		self.kernelMat[ind2, :] = row1;
		col1 = self.kernelMat[:, ind1];
		self.kernelMat[:, ind1] = self.kernelMat[:, ind2];
		self.kernelMat[:, ind2] = col1;

	def RemoveSupportVector(self, ind):
		# print("Removing SV: %d" % ind)
		self.svs[ind].x.refCount -= 1
		if (self.svs[ind].x.refCount == 0):
			# also remove support pattern if no more sv exists
			for i in range(len(self.sps)):
				if (self.sps[i] == self.svs[ind].x):
					del self.sps[i]
					break

		# make sure the support vector is at the back, this lets us keep the kernel matrix cached and valid
		if (ind < len(self.svs) - 1):
			self.SwapSupportVectors(ind, len(self.svs)-1)
			ind = len(self.svs) - 1
		del self.svs[ind]

	def Debug(self):
		print("%d/%d support patterns/vectors" % (len(self.sps), (self.svs)))
		self.UpdateDebugImage()
		cv2.imshow("learner", self.debugImage)

	def UpdateDebugImage(self):
		# self.debugImage.setTo(0)	# already all zero matrix
		n = len(self.svs)
		if (n == 0):
			return

		kCanvasSize = 600
		gridSize = int(math.sqrt(n-1)) + 1
		tileSize = int(kCanvasSize/gridSize)

		if (tileSize < 5):
			print("too many support vectors to display") 
			return

		temp = np.zeros((tileSize, tileSize), np.uint8)
		x = 0
		y = 0
		ind = 0
		vals = np.zeros(kMaxSVs, np.uint8)
		drawOrder = []

		for iset in range(2):
			for i in range(n):
				tmp = 1 if (iset == 0) else -1
				if (tmp * self.svs[i].b < 0.0):
					continue

				drawOrder.append(i)
				vals[ind] = self.svs[i].b
				ind += 1

				I = self.debugImage[y:y+tileSize, x:x+tileSize] # crop out the region
				# resize source image to be the same size as temp and store in temp
				cv2.resize(self.svs[i].x.images[self.svs[i].y], temp.shape, temp)
				# convert temp from grayscale to RGB and stores in I
				cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB, I)
				# draw rectangle
				w = 1.0
				color = (0, 255*w, 0) if (self.svs[i].b > 0.0) else (255*w, 0, 0)
				cv2.rectangle(I, (0, 0), (tileSize-1, tileSize-1), color, 3)

				x += tileSize
				if ((x+tileSize) > kCanvasSize):
					y += tileSize
					x = 0

		kKernelPixelSize = 2
		kernelSize = kKernelPixelSize * n

		kmin = self.kernelMat.min()
		kmax = self.kernelMat.max()

		if (kernelSize < self.debugImage.shape(1) and kernelSize < self.debugImage.shape(0)):
			K = self.debugImage[(self.debugImage.shape(1)-kernelSize) : self.debugImage.shape(1), (self.debugImage.shape(0)-kernelSize) : self.debugImage.shape(0)]
			for i in range(n):
				for j in range(n):
					Kij = K[j*kKernelPixelSize : j*kKernelPixelSize + kKernelPixelSize, i*kKernelPixelSize : i*kKernelPixelSize + kKernelPixelSize]
					v = 255 * (self.kernelMat[drawOrder[i], drawOrder[j]] - kmin) / (kmax - kmin)
					Kij[:] = (v, v, v)
		else:
			kernelSize = 0

		I = self.debugImage[:self.debugImage.shape(0)-200, self.debugImage.shape(1)-kernelSize:self.debugImage.shape(1)-kernelSize+200]
		I[:] = (255, 255, 255)
		
		# draw debug images
		II = I
		setGraphColor(0);
		drawGraph(vals, n, II, 0.0, 0.0, I.shape[1], I.shape[0]);



