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
import Kernels
from LaRank import LaRank

class Tracker:
    def __init__(self, config):
        self.config = config
        self.Reset()
 
    def Initialize(self, frame, rect):
        r = Rect()
        self.bb = r.initFromRect(rect)
        #image = IntegImg(frame) # orImgRep
        img = ImageRep(frame, True, False, False)
        self.UpdateLearner(img)
        self.initialized = True

    def IsInitialized(self):
        return self.initialized
         
    def GetBox(self):
        return self.bb

    def Reset(self):
        self.initialized = False
        self.needIntegImg = False
        self.bb = None
        self.pLearner = None
        self.features = []
        self.kernels = []
        featureCounts = []
        for feat in self.config.features:
            # TODO uncomment this once HaarFeatures is ready
            self.features.append(HaarFeatures(self.config))
            featureCounts.append(self.features[-1].GetCount())
            kerType = feat.kernelName
            if kerType == 'linear':
                self.kernels.append(Kernels.LinearKernel())
            elif kerType == 'gaussian':
                self.kernels.append(Kernels.GaussianKernel(feat.params[0]))
            elif kerType == 'intersection':
                self.kernels.append(Kernels.IntersectionKernel())
            elif kerType == 'chi2':
                self.kernels.append(Kernels.Chi2Kernel())
            # this should run for 1 iteration since we only use Haar Feature

        self.pLearner = LaRank(self.config, self.features[-1], self.kernels[-1])
        

    def Track(self, frame):
        #img = IntegImg(frame)
        img = ImageRep(frame, True, False, False)

        rects = Sampler.PixelSamples(self.bb, self.config.searchRadius)
        keptRects = []
        for rect in rects:
            if( rect.isInside(img.GetRect()) ):
                keptRects.append(rect)

        sample  = MultiSample(img, keptRects)
        scores = []
        self.pLearner.Eval(sample, scores)
        
        bestScore = max(scores)
        try:
            bestIndex = scores.index(bestScore)
        except ValueError:
            bestIndex = -1

        if not bestIndex == -1:
            self.bb = keptRects[bestIndex]
            UpdateLearner(img)

    def UpdateLearner(self, img):
        rects = Sampler.RadialSamples(self.bb, 2*self.config.searchRadius, 5, 16)
        keptRects = []
        keptRects.append(rects[0])
        for i, rect in enumerate(rects):
            if i < 1:
                continue
            if rect.IsInside(img.GetRect()) :
                keptRects.append(rect)
        sample = MultiSample(img, keptRects)
        self.pLearner.Update(sample, 0)

if __name__ == "__main__":
    c = Config("./config.txt")
    t = Tracker(c)
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
    initBB = Rect(100, 100, 80, 80)

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        if not t.IsInitialized():
            t.Initialize(frame, initBB)
        t.Track(frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    
    
