import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import sys
import random

class SupportPattern:
	# std::vector<Eigen::VectorXd> x; std::vector<FloatRect> yv; std::vector<cv::Mat> images; int y; int refCount;
	# def __init__(self, x, yv, images, y, refCount):
		# self.x = x
		# self.yv = yv
		# self.images = images
		# self.y = y
		# self.refCount = refCount
	def __init__(self):
		self.x = []
		self.yv = []
		self.images = []
		self.y = 0
		self.refCount = 0

class SupportVector:
	# SupportPattern* x; int y; double b; double g; cv::Mat image;
	def __init__(self, x, y, b, g, image):
		self.x = x
		self.y = y
		self.b = b
		self.g = g
		self.image = image

class LaRank:

	# 定义基本属性
	kMaxSVs = 2000
	kTileSize = 30
	
	m_sps = []	# SupportPattern vector/list
	m_svs = []	# SupportVector vector/list

	# (const Config& config, const Features& features, const Kernel& kernel)
	def __init__(self, conf, features, kernel):
		self.m_config = conf
		self.m_features = features
		self.m_kernel = kernel
		self.m_C = conf.svmC
		N = conf.svmBudgetSize + 2 if conf.svmBudgetSize > 0 else kMaxSVs	# int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;
		self.m_K = np.zeros((N, N), dtype=np.uint8)
		self.m_debugImage = np.zeros((800, 600, 3), np.uint8)
		self.m_C = conf.svmC

	# (const Eigen::VectorXd& x, const FloatRect& y)
	def Evaluate(self, x, y):
		f = 0
		for i in range(m_svs.size()):
			sv = m_svs[i]
			f += sv.b * self.m_kernel.Eval(x, sv.x.x[sv.y]) # Eval function in kernel class
		return f

	# (const MultiSample& sample, std::vector<double>& results)
	def Eval(self.sample, results): 
		centre = sample.GetRects()[0]
		self.m_features.Eval(sample, fvs)	# Eval function in Features, results in fvs variable
		results[:] = []
		for i in range(fvs.size()):
			y = sample.GetRects()[i]
			y.Translate(-centre.XMin(), -centre.YMin())	# functions in Rect
			results.append(Evaluate(fvs[i], y))

	# (const MultiSample& sample, int y)
	def Update(self, sample, y):
		sp = SupportPattern
		rects = sample.GetRects()	# GetRects function in Sample class
		center = rects[y]
		for i in range(rects.size()):
			r = rects[i]
			r.Translate(-center.XMin(), -center.YMin())	# Translate function in Rect class
			sp.yv.append(r)
			if (!self.m_config.quietMode and self.m_config.degugMode):
				im = np.zeros((kTileSize, kTileSize), np.uint8)
				# im = cv2.CreatMat((kTileSize, kTileSize), cv2.CV_8UC1)
				rect = rects[i]

				roi = [rect.XMin(), rect.XMin()+rect.Width(), rect.YMin(), rect.YMin()+rect.Height()] #[xmin, xmax, ymin, ymax]
				cv2.resize(sample.GetImage().GetImage(0)[roi[2]:roi[3], roi[0]:roi[1]], im.shape, im)
				# cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
				# cv::resize(sample.GetImage().GetImage(0)(roi), im, im.size());

				sp.images.extend(im)

		# evaluate feature for each sample
		sp.x.resize(rects.size())

		self.m_features.Eval(sample, sp.x)
		# const_cast<Features&>(m_features).Eval(sample, sp->x);

		sp.y = y
		sp.refCount = 0
		m_sps.extend(sp)

		ProcessNew(m_sps.size()-1)
		BudgetMaintenance()

		for i in range(10):
			Reprocess()
			BudgetMaintenance()

	def BudgetMaintenance(self):
		if (self.m_config.svmBudgetSize > 0):
			while (m_svs.size() > self.m_config.svmBudgetSize):
				BudgetMaintenanceRemove()

	def Reprocess():
		ProcessOld()
		for i in range(10):
			Optimize()

	# (const FloatRect& y1, const FloatRect& y2)
	def Loss(y1, y2):
		return 1 - y1.Overlap(y2)


	def ComputeDual(self):
		d = 0
		for i in range(m_svs.size()):
			sv = m_svs[i]
			d -= sv.b * Loss(sv.x.yv[sv.y], sv.x.yv[sv.x.y])
			for j in range(m_svs.size()):
				d -= 0.5 * sv.b * m_svs[j].b * self.m_K[i, j]
		return d

	# (int ipos, int ineg)
	def SMOStep(self, ipos, ineg):
		return if (ipos == ineg)
		svp = m_svs[ipos]
		svn = m_svs[ineg]
		assert (svp.x == svn.x)
		sp = svp.x

		# print("SMO: gpos: %d gneg:  " % (svp.g, svp.g))
		if ((svp->g - svn->g) < 1e-5):
			print("SMO: skipping")
		else:
			kii = self.m_K[ipos, ipos] + self.m_K[ineg, ineg] - 2 * self.m_K[ipos, ineg]
			lu = (svp.g - svn.g) / kii
			l = min(lu, self.m_C * (svp.y == sp.y) - svp.b)

			svp.b += l
			svn.b -= l

			# update gradient
			for i in range(m_svs.size()):
				svi = m_svs[i]
				svi.g -= l * (self.m_K[i, ipos] - self.m_K[i, ineg])
			# print("SMO: %d, %d -- %d, %d (%d)" % (ipos, ineg, svp.b, svn.b, l))

		# check if we should remove either sv now
		if (abs(svp.b) < 1e-8):
			RemoveSupportVector(ipos)
			if (ineg == m_svs.size()):
				# ineg and ipos will have been swapped during sv removal
				ineg = ipos

		if (abs(svn.b) < 1e-8):
			RemoveSupportVector(ineg)

	# (int ind)
	def MinGradient(self, ind):
		sp = m_sps[ind]
		minGrad = [-1, sys.float_info.max]
		for i in range(sp.yv.size()):
			grad = -Loss(sp.yv[i], sp.yv[sp.y]) - Evaluate(sp.x[i], sp.yv[i])
			if (grad < minGrad[1]):
				minGrad[0] = i
				minGrad[1] = grad
		return minGrad

	# (int ind)
	def ProcessNew(self, ind):
		# gradient is -f(x, y) since loss = 0
		yp = AddSupportVector(m_sps[ind], m_sps[ind].y, -Evaluate(m_sps[ind].x[m_sps[ind].y],m_sps[ind].yv[m_sps[ind].y]))

		minGrad = MinGradient(ind)
		yn = AddSupportVector(m_sps[ind], minGrad[0], minGrad[1])

		SMOStep(yp, yn)

	def ProcessOld(self):
		return if (m_sps.size() == 0)

		# choose pattern to process
		ind = random.randrange(m_sps.size())

		# find existing sv with largest grad and nonzero beta
		yp = -1
		maxGrad = -sys.float_info.max
		for i in range(m_svs.size()):
			continue if (m_svs[i].x != m_sps[ind])

			svi = m_svs[i]
			if (svi.g > maxGrad and svi.b < self.m_C * (svi.y == m_sps[ind].y)):
				yp = i
				maxGrad = svi.g

		assert (yp != -1)
		return if (yp == -1)

		# find potentially new sv with smallest grad
		minGrad = MinGradient(ind)
		yn = -1
		for i in range(m_svs.size()):
			continue if (m_svs[i].x != m_sps[ind])

			if (m_svs[i].y == minGrad[0]):
				yn = i
				break

		# add new sv
		yn = AddSupportVector(m_sps[ind], minGrad[0], minGrad[1]) if (yn == -1)

		SMOStep(yp, yn)

	def Optimize(self):
		return if (m_sps.size() == 0)

		# choose pattern to optimize
		ind = random.randrange(m_sps.size())

		yp = -1
		yn = -1
		maxGrad = -sys.float_info.max
		minGrad = sys.float_info.max
		for i in range(m_svs.size()):
			continue if (m_svs[i].x != m_sps[ind])

			svi = m_svs[i]
			if (svi.g > maxGrad and svi.b < self.m_C * (svi.y == m_sps[ind].y)):
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
		sv = SupportVector(0, x, y, g) # miss image here	################
		# SupportVector* sv = new SupportVector;
		# sv->b = 0.0;
		# sv->x = x;
		# sv->y = y;
		# sv->g = g;

		ind = m_svs.size()
		m_svs.extend(sv)
		x.refCount += 1

		# print("Adding SV: ", ind)

		# update kernel matrix
		for i in range(ind):
			self.m_K[i, ind] = self.m_kernel.Eval(m_svs[i].x-.x[m_svs[i].y], x.x[y])
			self.m_K[ind, i] = self.m_K[i, ind]
		
		self.m_K[ind, ind] = self.m_kernel.Eval(x.x[y])
		return ind

	def SwapSupportVectors(self, ind1, ind2):
		tmp = m_svs[ind1];
		m_svs[ind1] = m_svs[ind2];
		m_svs[ind2] = tmp;
		
		row1 = m_K[ind1, :];
		m_K[ind1, :] = m_K[ind2, :];
		m_K[ind2, :] = row1;
		
		col1 = m_K[:, ind1];
		m_K[:, ind1] = m_K[: ind2];
		m_K[:, ind2] = col1;

	def RemoveSupportVector(self, ind):
		# print("Removing SV: " ind)

		m_svs[ind].x.refCount -= 1
		if (m_svs[ind].x.refCount == 0):
			# also remove support pattern
			for i in range(m_sps.size()):
				if (m_sps[i] == m_svs[ind].x):
					del m_sps[i]
					break

		# make sure the support vector is at the back, this lets us keep the kernel matrix cached and valid
		if (ind < m_svs.size - 1):
			SwapSupportVectors(ind, m_svs.size()-1)
			ind = m_svs.size() - 1
		del m_svs[ind]
		del m_svs[-1]

	def BudgetMaintenanceRemove(self):
		# find negative sv with smallest effect on discriminant function if removed
		minVal = sys.float_info.max
		yn = -1
		yp = -1
		for i in range(m_svs.size()):
			if (m_svs[i].b < 0):
				# find corresponding postive sv
				j = -1

				for k in range(m_svs.size()):
					if (m_svs[k].b > 0 and m_svs[k].x == m_svs[i].x):
						j = k
						break
				
				val =m_svs[i].b * m_svs[i].b * (self.m_K[i, i] + self.m_K[j, j] - 2 * self.m_K[i, j])
				if (val < minVal):
					minVal = val
					yn = i
					yp = j

		# adjust weight of positive sv to compensate for removal of negative
		m_svs[yp].b += m_svs[yn].b

		# remove negative sv
		RemoveSupportVector(yn)
		# yp and yn will have been swapped during sv removal
		yp = yn if (yp == m_svs.size())
		# also remove positive sv
		RemoveSupportVector(yp) if (m_svs[yp].b < 1e-8)

		# update gradients
		# TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
		for i in range(m_svs.size()):
			svi = m_svs[i]
			svi.g = -Loss(svi.x.yv[svi.y], svi.x.yv[svi.x.y]) - Evaluate(svi.x.x[svi.y], svi.x.yv[svi.y])

	def Debug(self):
		print("%d/%d support patterns/vectors" % (m_sps.size(), m_svs.size()))
		UpdateDebugImage()
		cv2.imshow("learner", self.m_debugImage)

	def UpdateDebugImage(self):
		# self.m_debugImage.setTo(0)	# already all zero matrix
		
		n = m_svs.size()
		return if (n == 0)

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
				continue if (tmp * m_svs[i].b < 0)

				drawOrder.extend(i)
				vals[ind] = m_svs[i].b
				ind += 1

				I = self.m_debugImage[y:y+tileSize, x:x+tileSize] # crop out the region
				# resize source image to be the same size as temp and store in temp
				cv2.resize(m_svs[i].x.images[m_svs[i].y], temp.shape, temp)
				# convert temp from grayscale to RGB and stores in I
				cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB, I)
				# draw rectangle
				w = 1
				color = (0, 255*w, 0) if (m_svs[i].b > 0) else (255*w, 0, 0)
				cv2.rectangle(I, (0, 0), (tileSize-1, tileSize-1), color, 3)

				x += tileSize
				if ((x+tileSize) > kCanvasSize):
					y += tileSize
					x = 0

		kKernelPixelSize = 2
		kernelSize = kKernelPixelSize * n

		# Eigen function !!!!!!!!!!!!!!!!!!!!
		kmin = self.m_K.minCoeff()
		kmax = self.m_K.maxCoeff()
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
		II = I
		setGraphColor(0);
		drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!










