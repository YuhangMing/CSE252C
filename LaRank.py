import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import sys
import random

sys.path.append('./Config.py')
from Config import Config
sys.path.append('./HaarFeature.py')
from HaarFeature import HaarFeature
sys.path.append('./Kernels.py')
import Kernels 					# which kernel to import
sys.path.append('./Sample.py')
from Sample import Sample
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./GraphUtils.py')
import GraphUtils

class SupportPattern:
	# std::vector<Eigen::VectorXd> x; std::vector<FloatRect> yv; std::vector<cv::Mat> images; int y; int refCount;
	def __init__(self):
		self.x = []			# eigenVector
		self.yv = []		# floadRect
		self.images = []	# Mat
		self.y = 0			# int
		self.refCount = 0	# int

	def add_x(self, new_x):
		self.x.append(new_x)
	def add_yv(self, new_yv):
		self.yv.append(new_yv)
	def add_image(self, new_image):
		self.images.append(new_image)

class SupportVector:
	# SupportPattern* x; int y; double b; double g; cv::Mat image;
	def __init__(self, x=0.0, y=0.0, b=0.0, g=0.0, image = []):
		# x, y, b, g are definded in Update step, images are added in UpdateDebugImage step, use np.array to asign size directly
		self.x = x 			# supportPattern
		self.y = y 			# int, index of the rects in sp.yv
		self.b = b 			# double, bias
		self.g = g 			# double, value of the gradient(mini)
		self.image = image 	# Mat

class LaRank:

	kMaxSVs = 2000
	kTileSize = 30
	
	# m_sps = []	# SupportPattern vector/list
	# m_svs = []	# SupportVector vector/list

	# (const Config& config, const Features& features, const Kernel& kernel)
	def __init__(self, conf, features, kernel):
		self.m_config = config 									# Configuration class
		self.m_features = features 								# Feature class
		self.m_kernel = kernel 									# Kernel class
		self.m_C = conf.svmC 									# slack variable in svm
		N = conf.svmBudgetSize + 2 if conf.svmBudgetSize > 0 else kMaxSVs	# int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;
		self.m_K = np.zeros((N, N), dtype=np.uint8) 			# Kernal matrix
		self.m_debugImage = np.zeros((800, 600, 3), np.uint8) 	# sv images shown in learner window

		self.m_sps = []
		self.m_svs = []

	def add_sps(self, new_sp):
		self.m_sps.append(new_sp)
	def add_svs(self, new_sv):
		self.m_svs.append(new_sv)

	# (const Eigen::VectorXd& x, const FloatRect& y)
	def Evaluate(self, x, y):
		f = 0.0
		for i in range(len(self.m_svs)):
			sv = self.m_svs[i]
			f += sv.b * self.m_kernel.Eval(x, sv.x.x[sv.y]) # Eval function in kernel class
		return f

	# (const MultiSample& sample, std::vector<double>& results)
	def Eval(self, sample, results): 
		centre = sample.GetRects()[0]
		self.m_features.Eval(sample, fvs)	# Eval function in Features, results in fvs variable
		results[:] = []
		for i in range(len(fvs)):
			# express y in coord fram of center sample
			y = sample.GetRects()[i]
			y.Translate(-centre.XMin(), -centre.YMin())	# functions in Rect
			results.append(Evaluate(fvs[i], y))

	# (const MultiSample& sample, int y)
	def Update(self, sample, y):
		sp = SupportPattern()
		rects = sample.GetRects()	# GetRects function in Sample class, should be a list of rects, 4xn
		center = rects[y]
		for i in range(len(rects)):
			r = rects[i]
			r.Translate(-center.XMin(), -center.YMin())	# Translate function in Rect class
			sp.add_yv(r)
			if (not(self.m_config.quietMode) and self.m_config.degugMode):
				im = np.zeros((kTileSize, kTileSize), np.uint8)
				# im = cv2.CreatMat((kTileSize, kTileSize), cv2.CV_8UC1)
				rect = rects[i]

				roi = [rect.XMin(), rect.XMin()+rect.Width(), rect.YMin(), rect.YMin()+rect.Height()] #[xmin, xmax, ymin, ymax]
				cv2.resize(sample.GetImage().GetImage(0)[roi[2]:roi[3], roi[0]:roi[1]], im.shape, im)
				# cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
				# cv::resize(sample.GetImage().GetImage(0)(roi), im, im.size());

				sp.add_image(im)

		# evaluate feature for each sample
		sp.x.resize(len(rects))		# may not need this resize here, try get value one by one and append/add to list
		self.m_features.Eval(sample, sp.x)
		# const_cast<Features&>(m_features).Eval(sample, sp->x);

		sp.y = y
		sp.refCount = 0
		self.add_sps(sp) # self.m_sps.extend(sp)

		ProcessNew( len(self.m_sps)-1 )
		BudgetMaintenance()

		for i in range(10):
			Reprocess()
			BudgetMaintenance()

	def BudgetMaintenance(self):
		if (self.m_config.svmBudgetSize > 0):
			while (len(self.m_svs) > self.m_config.svmBudgetSize):
				BudgetMaintenanceRemove()

	def Reprocess():
		ProcessOld()
		for i in range(10):
			Optimize()

	# (const FloatRect& y1, const FloatRect& y2)
	def Loss(y1, y2):
		return 1 - y1.Overlap(y2)


	def ComputeDual(self):
		d = 0.0
		for i in range(len(self.m_svs)):
			sv = self.m_svs[i]
			d -= sv.b * Loss(sv.x.yv[sv.y], sv.x.yv[sv.x.y])
			for j in range(len(self.m_svs)):
				d -= 0.5 * sv.b * self.m_svs[j].b * self.m_K[i, j]
		return d

	# (int ipos, int ineg)
	def SMOStep(self, ipos, ineg):
		if (ipos == ineg):
			return 
		svp = self.m_svs[ipos]
		svn = self.m_svs[ineg]
		assert (svp.x == svn.x)
		sp = svp.x

		# print("SMO: gpos: %d gneg:  " % (svp.g, svp.g))
		if ((svp.g - svn.g) < 1e-5):
			print("SMO: skipping")
		else:
			kii = self.m_K[ipos, ipos] + self.m_K[ineg, ineg] - 2 * self.m_K[ipos, ineg]
			lu = (svp.g - svn.g) / kii
			# no need to clamp against 0 since we'd have skipped in that case
			l = min(lu, self.m_C * (svp.y == sp.y) - svp.b)

			svp.b += l
			svn.b -= l

			# update gradient
			for i in range(len(self.m_svs)):
				svi = self.m_svs[i]
				svi.g -= l * (self.m_K[i, ipos] - self.m_K[i, ineg])
			# print("SMO: %d, %d -- %d, %d (%d)" % (ipos, ineg, svp.b, svn.b, l))

		# check if we should remove either sv now
		if (abs(svp.b) < 1e-8):
			RemoveSupportVector(ipos)
			if (ineg == len(self.m_svs)):
				# ineg and ipos will have been swapped during sv removal
				ineg = ipos

		if (abs(svn.b) < 1e-8):
			RemoveSupportVector(ineg)

	# (int ind)
	def MinGradient(self, ind):
		sp = self.m_sps[ind]
		minGrad = [-1, sys.float_info.max]
		for i in range(len(sp.yv)):
			grad = -Loss(sp.yv[i], sp.yv[sp.y]) - Evaluate(sp.x[i], sp.yv[i])
			if (grad < minGrad[1]):
				minGrad[0] = i
				minGrad[1] = grad
		return minGrad

	# (int ind)
	def ProcessNew(self, ind):
		# gradient is -F(x, y) since loss = 0
		yp = AddSupportVector(self.m_sps[ind], self.m_sps[ind].y, -Evaluate( self.m_sps[ind].x[ self.m_sps[ind].y ], self.m_sps[ind].yv[ self.m_sps[ind].y ] ))

		minGrad = MinGradient(ind)
		yn = AddSupportVector(self.m_sps[ind], minGrad[0], minGrad[1])

		SMOStep(yp, yn)

	def ProcessOld(self):
		if (len(self.m_sps) == 0):
			return

		# choose pattern to process
		ind = random.randrange(len(self.m_sps))

		# find existing sv with largest grad and nonzero beta
		yp = -1
		maxGrad = -sys.float_info.max
		for i in range(len(self.m_svs)):
			if (self.m_svs[i].x != self.m_sps[ind]):
				continue

			svi = self.m_svs[i]
			if (svi.g > maxGrad and svi.b < self.m_C * (svi.y == self.m_sps[ind].y)):
				yp = i
				maxGrad = svi.g

		assert (yp != -1)
		if (yp == -1):
			return 

		# find potentially new sv with smallest grad
		minGrad = MinGradient(ind)
		yn = -1
		for i in range(len(self.m_svs)):
			if (self.m_svs[i].x != self.m_sps[ind]):
				continue 

			if (self.m_svs[i].y == minGrad[0]):
				yn = i
				break

		# add new sv
		if (yn == -1):
			yn = AddSupportVector(self.m_sps[ind], minGrad[0], minGrad[1])

		SMOStep(yp, yn)

	def Optimize(self):
		if (len(self.m_sps) == 0):
			return

		# choose pattern to optimize
		ind = random.randrange(len(self.m_sps))

		yp = -1
		yn = -1
		maxGrad = -sys.float_info.max
		minGrad = sys.float_info.max
		for i in range(len(self.m_svs)):
			if (self.m_svs[i].x != self.m_sps[ind]):	# search among the support patterns
				continue

			svi = self.m_svs[i]
			if (svi.g > maxGrad and svi.b < self.m_C * (svi.y == self.m_sps[ind].y)):
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

		SMOStep(yp, yn)

	def AddSupportVector(self, x, y, g):
		sv = SupportVector(0.0, x, y, g) 
		# SupportVector* sv = new SupportVector;
		# sv->b = 0.0;
		# sv->x = x;
		# sv->y = y;
		# sv->g = g;

		ind = len(self.m_svs)
		self.add_svs(sv)
		x.refCount += 1

		# print("Adding SV: ", ind)

		# update kernel matrix
		for i in range(ind):
			self.m_K[i, ind] = self.m_kernel.Eval(self.m_svs[i].x.x[ self.m_svs[i].y ], x.x[y])
			self.m_K[ind, i] = self.m_K[i, ind]
		
		self.m_K[ind, ind] = self.m_kernel.Eval(x.x[y])
		return ind

	def SwapSupportVectors(self, ind1, ind2):
		tmp = self.m_svs[ind1];
		self.m_svs[ind1] = self.m_svs[ind2];
		self.m_svs[ind2] = tmp;
		
		row1 = self.m_K[ind1, :];
		self.m_K[ind1, :] = self.m_K[ind2, :];
		self.m_K[ind2, :] = row1;
		
		col1 = self.m_K[:, ind1];
		self.m_K[:, ind1] = self.m_K[: ind2];
		self.m_K[:, ind2] = col1;

	def RemoveSupportVector(self, ind):
		# print("Removing SV: " ind)

		self.m_svs[ind].x.refCount -= 1
		if (self.m_svs[ind].x.refCount == 0):
			# also remove support pattern
			for i in range(len(self.m_sps)):
				if (self.m_sps[i] == self.m_svs[ind].x):
					del self.m_sps[i]
					break

		# make sure the support vector is at the back, this lets us keep the kernel matrix cached and valid
		if (ind < self.m_svs.size - 1):
			SwapSupportVectors(ind, len(self.m_svs)-1)
			ind = len(self.m_svs) - 1
		del self.m_svs[ind]
		del self.m_svs[-1]

	def BudgetMaintenanceRemove(self):
		# find negative sv with smallest effect on discriminant function if removed
		minVal = sys.float_info.max
		yn = -1
		yp = -1
		for i in range(len(self.m_svs)):
			if (self.m_svs[i].b < 0.0):
				# find corresponding postive sv
				j = -1

				for k in range(len(self.m_svs)):
					if (self.m_svs[k].b > 0.0 and self.m_svs[k].x == self.m_svs[i].x):
						j = k
						break
				
				val =self.m_svs[i].b * self.m_svs[i].b * (self.m_K[i, i] + self.m_K[j, j] - 2.0 * self.m_K[i, j])
				if (val < minVal):
					minVal = val
					yn = i
					yp = j

		# adjust weight of positive sv to compensate for removal of negative
		self.m_svs[yp].b += self.m_svs[yn].b

		# remove negative sv
		RemoveSupportVector(yn)
		# yp and yn will have been swapped during sv removal
		if (yp == len(self.m_svs)):
			yp = yn
		# also remove positive sv
		if (self.m_svs[yp].b < 1e-8):
			RemoveSupportVector(yp)

		# update gradients
		# TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
		for i in range(len(self.m_svs)):
			svi = self.m_svs[i]
			svi.g = -Loss(svi.x.yv[svi.y], svi.x.yv[svi.x.y]) - Evaluate(svi.x.x[svi.y], svi.x.yv[svi.y])

	def Debug(self):
		print("%d/%d support patterns/vectors" % (len(self.m_sps), (self.m_svs)))
		UpdateDebugImage()
		cv2.imshow("learner", self.m_debugImage)

	def UpdateDebugImage(self):
		# self.m_debugImage.setTo(0)	# already all zero matrix
		
		n = len(self.m_svs)
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
		# or drawOrder = np.zeros(kMaxSVs, np.uint8)

		for iset in range(2):
			for i in range(n):
				tmp = 1 if (iset == 0) else -1
				if (tmp * self.m_svs[i].b < 0.0):
					continue

				drawOrder.append(i)
				vals[ind] = self.m_svs[i].b
				ind += 1

				I = self.m_debugImage[y:y+tileSize, x:x+tileSize] # crop out the region
				# resize source image to be the same size as temp and store in temp
				cv2.resize(self.m_svs[i].x.images[self.m_svs[i].y], temp.shape, temp)
				# convert temp from grayscale to RGB and stores in I
				cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB, I)
				# draw rectangle
				w = 1.0
				color = (0, 255*w, 0) if (self.m_svs[i].b > 0.0) else (255*w, 0, 0)
				cv2.rectangle(I, (0, 0), (tileSize-1, tileSize-1), color, 3)

				x += tileSize
				if ((x+tileSize) > kCanvasSize):
					y += tileSize
					x = 0

		kKernelPixelSize = 2
		kernelSize = kKernelPixelSize * n

		
		# kmin = self.m_K.minCoeff()
		kmin = self.m_K.min()
		# kmax = self.m_K.maxCoeff()
		kmax = self.m_K.max()
		

		if (kernelSize < self.m_debugImage.shape(1) and kernelSize < self.m_debugImage.shape(0)):
			K = self.m_debugImage[(self.m_debugImage.shape(1)-kernelSize) : self.m_debugImage.shape(1), (self.m_debugImage.shape(0)-kernelSize) : self.m_debugImage.shape(0)]
			for i in range(n):
				for j in range(n):
					Kij = K[j*kKernelPixelSize : j*kKernelPixelSize + kKernelPixelSize, i*kKernelPixelSize : i*kKernelPixelSize + kKernelPixelSize]
					v = 255 * (self.m_K[drawOrder[i], drawOrder[j]] - kmin) / (kmax - kmin)
					Kij[:] = (v, v, v)
		else:
			kernelSize = 0

		I = self.m_debugImage[:self.m_debugImage.shape(0)-200, self.m_debugImage.shape(1)-kernelSize:self.m_debugImage.shape(1)-kernelSize+200]
		I[:] = (255, 255, 255)
		
		# GraphUtils.cpp !!!!!!!!!!!!!!!!!!!!
		# incorporate with GraphUtils later
		# II = I
		# setGraphColor(0);
		# drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







