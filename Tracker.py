#!/usr/bin/env python
import numpy as np
import cv2

from Rect import Rect
from Config import Config
from ImageRep import ImageRep
from Sampler import Sampler
from Sample import Sample
from Sample import MultiSample
from HaarFeatures import HaarFeatures
from RawFeatures import RawFeatures
from HistogramFeatures import HistogramFeatures
from MultiFeatures import MultiFeatures
import Kernels
from LaRank import LaRank

# Tracker class
# Puts all the pieces together
class Tracker:
    def __init__(self, config):
        self.config = config
        self.Reset()
 
    # Initialize used for the initial frame
    def Initialize(self, frame, rect):
        self.box = Rect()
        self.box.initFromRect(rect)
        img = ImageRep(frame, self.needIntegImg, self.needIntegHist, False)
        self.UpdateLearner(img)
        self.initialized = True

    # check if the tracker is already initialized
    def IsInitialized(self):
        return self.initialized
         
    # returns the current position of the bound box
    def GetBox(self):
        return self.box

    # Add features
    def AddFeature(self, fname):
        if fname == 'haar':
            self.features.append(HaarFeatures(self.config))
            self.needIntegImg = True
        elif fname == 'raw':
            self.features.append(RawFeatures(self.config))
        elif fname == 'histogram':
            self.needIntegHist = True
            self.features.append(HistogramFeatures(self.config));
        
    # Add kernel
    def AddKernel(self, kname, feature):
        if kname == 'linear':
            self.kernels.append(Kernels.LinearKernel())
        elif kname == 'gaussian':
            self.kernels.append(Kernels.GaussianKernel(feature.params[0]))
        elif kname == 'intersection':
            self.kernels.append(Kernels.IntersectionKernel())
        elif kname == 'chi2':
            self.kernels.append(Kernels.Chi2Kernel())
        
    # reset, useful if the tracker is not initialized properly
    def Reset(self):
        self.initialized = False
        self.box = None
        self.features = []
        self.kernels = []
        self.needIntegImg = False
        self.needIntegHist = False
        self.learner = None
        
	# keep a list of feature counts
        featureCounts = []

        # check for number of features in the config file
        # should only run for 1 iteration for our experiemnt (Haar only) 
        for feat in self.config.features:
            self.AddFeature(feat.featureName)
            self.AddKernel(feat.kernelName, feat)
            featureCounts.append(self.features[-1].GetCount())

        # use combined feature/kernel when there are multiple
        if (len(self.config.features) > 1):
            self.features.append(MultiFeatures(self.features))
            self.kernels.append(Kernels.MultiKernel(self.kernels, featureCounts))

        self.learner = LaRank(self.config, self.features[-1], self.kernels[-1])
        
    # Track the object of interest
    def Track(self, frame):
        img = ImageRep(frame, self.needIntegImg, self.needIntegHist, False)
	
	# dense sampling for tracking process
        s = Sampler()
        sampled_rects = s.PixelSamples(self.box, self.config.searchRadius)

	# check if the box is still within the image
        usable_rects = []
        for rect in sampled_rects:
            if( rect.isInside(img.GetRect()) ):
                usable_rects.append(rect)

	# Make multiSample object 
	# This step is really expensive for some reason 
        # TODO optimize if possible
        msample  = MultiSample(img, usable_rects)
        scores = []
        self.learner.Eval(msample, scores)
        
        # find the best box
        best_score = max(scores)
        try:
            bestIndex = scores.index(best_score)
        except ValueError:
            bestIndex = -1

	# Update bounding box with the with best box
        if not bestIndex == -1:
            self.box = usable_rects[bestIndex]
            self.UpdateLearner(img)

    # Update Learner with new support vectors
    def UpdateLearner(self, img):
        # sparse sampling for unpdating Learner
        s = Sampler()
        sampled_rects = s.RadialSamples(self.box, 2*self.config.searchRadius, 5, 16)

	# rect to keep
        usable_rects = []
        for i, rect in enumerate(sampled_rects):
            if i < 1:
                #always keep the original frame
                usable_rects.append(sampled_rects[0])
            elif rect.isInside(img.GetRect()) :
                # make sure other sampled frames are inside the image
                usable_rects.append(rect)
	
        msample = MultiSample(img, usable_rects)
        self.learner.Update(msample, 0)


'''
if __name__ == "__main__":
    c = Config("./config.txt")
    t = Tracker(c)

    cv2.namedWindow("preview")
    # cv2.namedWindow("test")
    # vc = cv2.VideoCapture(0)

    # if vc.isOpened(): # try to get the first frame
    #     rval, frame = vc.read()
    # else:
    #     rval = False

    # initBB = Rect(100, 100, 80, 80)

    # while rval:
    #     cv2.imshow("preview", frame)
    #     rval, frame = vc.read()
    #     frame = cv2.resize(frame, (c.frameWidth, c.frameHeight))
    #     if t.IsInitialized():
    #         t.Track(frame)
    #     key = cv2.waitKey(20)
    #     if key == 32:
    #         t.Initialize(frame, initBB)
            
    #     if key == 27: # exit on ESC
    #         break
    # cv2.destroyWindow("preview")
 
    tmp = cv2.imread('./data/Girl/img/%04d.jpg' % 1)
    scaleW = float(c.frameWidth) / tmp.shape[1]
    scaleH = float(c.frameHeight) / tmp.shape[0]
    initBB = Rect(int(57 * scaleW), int(21 * scaleH), int(31 * scaleW), int(45 * scaleH))
    initBB.printStr()
    for i in range(1, 501):
        frame = cv2.imread('./data/Girl/img/%04d.jpg' % i, 0)
        print(i)
        # print((c.frameWidth, c.frameHeight))
        # cv2.imshow("test", frame)
        frame = cv2.resize(frame, (c.frameWidth, c.frameHeight))
        result = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # cv2.imshow("preview", frame)
        if t.IsInitialized():
            t.Track(frame)
            newBB = t.GetBox()
            pt1 = (newBB.XMin(), newBB.YMin())
            pt2 = (newBB.XMax(), newBB.YMax())
            # cv2.rectangle(frame, pt1, pt2, (0, 0, 255))
            cv2.rectangle(result, pt1, pt2, (0, 0, 255))
            cv2.imshow("preview", result)
            # t.Track(frame)
        else:
            pt1 = (initBB.XMin(), initBB.YMin())
            pt2 = (initBB.XMax(), initBB.YMax())
            # cv2.rectangle(frame, pt1, pt2, (0, 0, 255))
            cv2.rectangle(result, pt1, pt2, (0, 0, 255))
            cv2.imshow("preview", result)
            t.Initialize(frame, initBB)
        uBB = t.GetBox()
        print(uBB.XMin(), uBB.YMin(), uBB.Width(), uBB.Height())
        # t.Initialize(frame, initBB)
        key = cv2.waitKey(20)
            
            
        # if key == 27: # exit on ESC
        #     break   
    # cv2.waitKey(0)
    cv2.destroyWindow("preview")
    
''' 
