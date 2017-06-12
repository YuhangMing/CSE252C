import numpy as np
np.set_printoptions(threshold=np.nan)
# import matplotlib.pyplot as plt
import math
import cv2
import sys
import random

sys.path.append('./Config.py')
from Config import Config
# sys.path.append('./HaarFeatures.py')
# from HaarFeatures import HaarFeatures
sys.path.append('./Features.py')
from Features import Features
sys.path.append('./Kernels.py')
from Kernels import GaussianKernel	# which kernel to import
sys.path.append('./Sample.py')
from Sample import Sample
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./GraphUtils.py')
import GraphUtils

###### DEBUG MODE ########
sys.path.append('./ImageRep.py')
from ImageRep import ImageRep
sys.path.append('./Sampler.py')
from Sampler import Sampler
sys.path.append('./Sample.py')
from Sample import MultiSample
###### DEBUG MODE ########

class SupportPattern:
	# std::vector<Eigen::VectorXd> x; std::vector<FloatRect> yv; std::vector<cv::Mat> images; int y; int refCount;
	def __init__(self):
		self.x = []			# eigenVector, value of features
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

kMaxSVs = 2000
kTileSize = 30

class LaRank:

	# kMaxSVs = 2000
	# kTileSize = 30
	
	# m_sps = []	# SupportPattern vector/list
	# m_svs = []	# SupportVector vector/list

	# (const Config& config, const Features& features, const Kernel& kernel)
	def __init__(self, conf, features, kernel):
		self.m_config = conf 									# Configuration class
		self.m_features = features 								# Feature class
		self.m_kernel = kernel 									# Kernel class
		self.m_C = float(conf.svmC) 							# slack variable in svm
		N = conf.svmBudgetSize + 2 if conf.svmBudgetSize > 0 else kMaxSVs	# int N = conf.svmBudgetSize > 0 ? conf.svmBudgetSize+2 : kMaxSVs;
		self.m_K = np.zeros((N, N), dtype=np.float32) 			# Kernal matrix
		self.m_debugImage = np.zeros((800, 600, 3), np.uint8) 	# sv images shown in learner window

		self.m_sps = []
		self.m_svs = []

	def add_sps(self, new_sp):
		self.m_sps.append(new_sp)
	def add_svs(self, new_sv):
		self.m_svs.append(new_sv)

	# (const Eigen::VectorXd& x, const FloatRect& y)
	def Evaluate(self, x, y):
		# print(x)
		# print("")
		# print(y)
		f = 0.0
		for i in range(len(self.m_svs)):
			sv = self.m_svs[i]
			# print(len(x))
			f += sv.b * self.m_kernel.Eval(x, sv.x.x[sv.y]) # Eval function in kernel class
		return f

	# (const MultiSample& sample, std::vector<double>& results)
	def Eval(self, sample, results): 
		centre = Rect()
		centre.initFromRect(sample.GetRects()[0])
		fvs = np.zeros((len(sample.GetRects()), 192), np.float32)
		self.m_features.Eval(sample, fvs)	# Eval function in Features, results in fvs variable
		results[:] = []
		for i in range(len(fvs)):
			# express y in coord fram of center sample

			y = Rect()
			y.initFromRect( sample.GetRects()[i] )
			y.Translate(-centre.XMin(), -centre.YMin())	# functions in Rect
			results.append(self.Evaluate(fvs[i], y))

	# (const MultiSample& sample, int y)
	def Update(self, sample, y):
		# print('start updating')
		sp = SupportPattern()
		rects = sample.GetRects()	# GetRects function in Sample class, should be a list of rects, 4xn
		center = Rect()
		center.initFromRect(rects[y])
		for i in range(len(rects)):
			r = Rect()
			r.initFromRect(rects[i])
			r.Translate(-center.XMin(), -center.YMin())	# Translate function in Rect class
			sp.add_yv(r)
			if (not(self.m_config.quietMode) and self.m_config.debugMode):
				im = np.zeros((kTileSize, kTileSize), np.uint8)
				# im = cv2.CreatMat((kTileSize, kTileSize), cv2.CV_8UC1)
				rect = Rect()
				rect.initFromRect(rects[i])

				roi = [rect.XMin(), rect.XMin()+rect.Width(), rect.YMin(), rect.YMin()+rect.Height()] #[xmin, xmax, ymin, ymax]
				cv2.resize(sample.GetImage().GetImage(0)[int(roi[2]):int(roi[3]), int(roi[0]):int(roi[1])], im.shape, im)
				# cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
				# cv::resize(sample.GetImage().GetImage(0)(roi), im, im.size());

				sp.add_image(im)

		# evaluate feature for each sample
		# sp.x.resize(len(rects))		# may not need this resize here, try get value one by one and append/add to list
		# sp.x = np.zeros((len(sample.GetRects()), 192), np.float32)
		
		# Print some vals ##################
		# print("LaRank.Update, entering haarfeatures.Eval: ")
		####################################

		self.m_features.Eval(sample, sp.x)	# const_cast<Features&>(m_features).Eval(sample, sp->x);

		# Print some vals ##################
		# print("before precess new")
		# print(sp.x[0])
		####################################

		sp.y = y
		sp.refCount = 0
		self.add_sps(sp)
		# self.m_sps.extend(sp)

		self.ProcessNew( len(self.m_sps)-1 )
		# print("after process new: ")
		# print(self.m_K)	
		self.BudgetMaintenance()

		for i in range(10):
			# print(i)
			# print(len(self.m_svs))
			self.Reprocess()
			self.BudgetMaintenance()

	def BudgetMaintenance(self):
		if (self.m_config.svmBudgetSize > 0):
			while (len(self.m_svs) > self.m_config.svmBudgetSize):
				self.BudgetMaintenanceRemove()

	def Reprocess(self):
		self.ProcessOld()
		for i in range(10):
			self.Optimize()

	# (const FloatRect& y1, const FloatRect& y2)
	def Loss(self, y1, y2):
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
			# print("SMO: skipping")
			pass
		else:
			kii = self.m_K[ipos, ipos] + self.m_K[ineg, ineg] - 2 * self.m_K[ipos, ineg]
			lu = (svp.g - svn.g) / kii
			# no need to clamp against 0 since we'd have skipped in that case
			l = min(lu, self.m_C * float(svp.y == sp.y) - svp.b)

			svp.b += l
			svn.b -= l

			# update gradient
			for i in range(len(self.m_svs)):
				svi = self.m_svs[i]
				svi.g -= l * (self.m_K[i, ipos] - self.m_K[i, ineg])
			# print("SMO: %d, %d -- %f, %f (%f)" % (ipos, ineg, svp.b, svn.b, l))

		# check if we should remove either sv now
		if (abs(svp.b) < 1e-8):
			self.RemoveSupportVector(ipos)
			if (ineg == len(self.m_svs)):
				# ineg and ipos will have been swapped during sv removal
				ineg = ipos

		if (abs(svn.b) < 1e-8):
			self.RemoveSupportVector(ineg)

	# (int ind)
	def MinGradient(self, ind):
		sp = self.m_sps[ind]
		minGrad = [-1, sys.float_info.max]
		for i in range(len(sp.yv)):
			grad = -self.Loss(sp.yv[i], sp.yv[sp.y]) - self.Evaluate(sp.x[i], sp.yv[i])
			if (grad < minGrad[1]):
				minGrad[0] = i
				minGrad[1] = grad
		return minGrad

	# (int ind)
	def ProcessNew(self, ind):
		# gradient is -F(x, y) since loss = 0
		# print("process new ind = ", ind)
		# AddSupportVector(self, x, y, g):
		print("length of x")
		print(len(self.m_sps[ind].x))
		print(self.m_sps[ind].y)
		yp = self.AddSupportVector(self.m_sps[ind], self.m_sps[ind].y, -self.Evaluate( self.m_sps[ind].x[ self.m_sps[ind].y ], self.m_sps[ind].yv[ self.m_sps[ind].y ] ))

		minGrad = self.MinGradient(ind)
		yn = self.AddSupportVector(self.m_sps[ind], minGrad[0], minGrad[1])

		self.SMOStep(yp, yn)

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
			if (svi.g > maxGrad and svi.b < self.m_C * float(svi.y == self.m_sps[ind].y)):
				yp = i
				maxGrad = svi.g

		assert (yp != -1)
		if (yp == -1):
			return 

		# find potentially new sv with smallest grad
		minGrad = self.MinGradient(ind)
		yn = -1
		for i in range(len(self.m_svs)):
			if (self.m_svs[i].x != self.m_sps[ind]):
				continue 

			if (self.m_svs[i].y == minGrad[0]):
				yn = i
				break

		# add new sv
		if (yn == -1):
			yn = self.AddSupportVector(self.m_sps[ind], minGrad[0], minGrad[1])

		self.SMOStep(yp, yn)

	def Optimize(self):
		if (len(self.m_sps) == 0):
			return

		# choose pattern to optimize
		ind = random.randrange(len(self.m_sps))

		yp = -1
		yn = -1
		maxGrad = -sys.float_info.max
		minGrad = sys.float_info.max

		# print(len(self.m_svs))

		for i in range(len(self.m_svs)):
			if (self.m_svs[i].x != self.m_sps[ind]):	# search among the support patterns
				continue

			svi = self.m_svs[i]
			if (svi.g > maxGrad and svi.b < self.m_C * float(svi.y == self.m_sps[ind].y)):
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

	def AddSupportVector(self, x, y, g):
		sv = SupportVector(x, y, 0.0, g) 
		# SupportVector* sv = new SupportVector;
		# sv->x = x;
		# sv->y = y;		index of the rects in sp.yv
		# sv->b = 0.0;
		# sv->g = g;		value of the gradient(mini)

		ind = len(self.m_svs)
		self.add_svs(sv)
		x.refCount += 1
		# print("Adding SV: ", ind)
		# print('y = %d, g = %f' % (y, g))

		# update kernel matrix
		for i in range(ind):
			self.m_K[i, ind] = self.m_kernel.Eval(self.m_svs[i].x.x[ self.m_svs[i].y ], x.x[y])
			self.m_K[ind, i] = self.m_K[i, ind]
		
		self.m_K[ind, ind] = self.m_kernel.Eval( x.x[y] )
		return ind

	def SwapSupportVectors(self, ind1, ind2):
		tmp = self.m_svs[ind1];
		self.m_svs[ind1] = self.m_svs[ind2];
		self.m_svs[ind2] = tmp;
		
		row1 = self.m_K[ind1, :];
		self.m_K[ind1, :] = self.m_K[ind2, :];
		self.m_K[ind2, :] = row1;
		
		col1 = self.m_K[:, ind1];
		self.m_K[:, ind1] = self.m_K[:, ind2];
		self.m_K[:, ind2] = col1;

	def RemoveSupportVector(self, ind):
		# print("Removing SV: %d" % ind)
		self.m_svs[ind].x.refCount -= 1
		if (self.m_svs[ind].x.refCount == 0):
			# also remove support pattern
			for i in range(len(self.m_sps)):
				if (self.m_sps[i] == self.m_svs[ind].x):
					del self.m_sps[i]
					break

		# make sure the support vector is at the back, this lets us keep the kernel matrix cached and valid
		if (ind < len(self.m_svs) - 1):
			self.SwapSupportVectors(ind, len(self.m_svs)-1)
			ind = len(self.m_svs) - 1
		del self.m_svs[ind]
		# del self.m_svs[-1]

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
		self.RemoveSupportVector(yn)
		# yp and yn will have been swapped during sv removal
		if (yp == len(self.m_svs)):
			yp = yn
		# also remove positive sv
		if (self.m_svs[yp].b < 1e-8):
			self.RemoveSupportVector(yp)

		# update gradients
		# TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
		for i in range(len(self.m_svs)):
			svi = self.m_svs[i]
			svi.g = -self.Loss(svi.x.yv[svi.y], svi.x.yv[svi.x.y]) - self.Evaluate(svi.x.x[svi.y], svi.x.yv[svi.y])

	def Debug(self):
		print("%d/%d support patterns/vectors" % (len(self.m_sps), (self.m_svs)))
		self.UpdateDebugImage()
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
		
		# draw debug images
		II = I
		setGraphColor(0);
		# drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
		drawGraph(vals, n, II, 0.0, 0.0, I.shape[1], I.shape[0]);


###### DEBUG MODE ########
if __name__ == '__main__':
	# set parameters
	# configPath = './config.txt'
	conf = Config('./config.txt')

	# print(len(conf.features))
	# print(conf.svmC)
	# print(conf.svmBudgetSize)
	main_features = []
	main_kernels = []
	main_featureCount = []
	numFeatures = len(conf.features)
	for i in range(numFeatures):
		main_features.append(Features(conf))
		main_kernels.append(GaussianKernel(conf.features[i].params[0]))
		main_featureCount.append(main_features[-1].GetCount())

	# debug LaRank
	# initialize
	m_pLearner = LaRank(conf, main_features[-1], main_kernels[-1])

	# update
	BB = [[57, 21, 31, 45], [58, 22, 31, 43], [60, 23, 29, 42], [61, 18, 31, 47], [61, 19, 35, 46],
		[67, 16, 30, 49], [67, 16, 36, 47], [69, 15, 38, 49], [73, 17, 36, 47], [74, 15, 39, 50]]
	for i in range(1, 11):
		print('image %d' % i)
		frame = cv2.imread('./%04d.jpg' % i)
		image = ImageRep(frame, True, False, True)

		iniBB = BB[i-1]
		m_bb = Rect()
		m_bb.initFromList(iniBB)
		# print(m_bb.XMin(), m_bb.YMin(), m_bb.Width(), m_bb.Height())
		sampler = Sampler()
		rects = sampler.RadialSamples(m_bb, 2*conf.searchRadius, 5, 16)
		keptRects = []
		keptRects.append(rects[0])
		# print(rects[0].XMin(), rects[0].YMin(), rects[0].Width(), rects[0].Height())
		# for j in range(0, len(rects)):
		# 	print(rects[j].XMin(), rects[j].YMin(), rects[j].Width(), rects[j].Height())
		for j in range(1, len(rects)):
			# print(rects[j].XMin(), rects[j].YMin(), rects[j].Width(), rects[j].Height())
			if ( not( rects[j].IsInside(image.GetRect()) ) ): 
				# print('not inside')
				continue
			keptRects.append(rects[j]);
			# print('inside')
		# print('# of inside rects: '+ str(len(keptRects)))
		multi_sample = MultiSample(image, keptRects)
		m_pLearner.Update(multi_sample, 0);
###### DEBUG MODE ########



