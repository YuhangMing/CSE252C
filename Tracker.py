#!/usr/bin/env python
import numpy as np
import cv2

import Config
import ImageRep
import Sampler
import Sample
import HaarFeatures
import Kernels
import LaRank

class Tracker:
    def __init__(self, config):
        self.config = config
        self.Reset()
 
    def Initialize(self, frame, rect);
        r = Rect()
        self.bb = r.initializeFromRect(rect)
        #image = IntegImg(frame) # orImgRep
        img = ImageRep(frame)
        self.UpdateLearner(img)
        self.initialised = True

    def IsInitialized(self):
        return self.initilised
         
    def GetBox(self):
        return self.box

    def Reset(self):
        self.initialized = False
        self.needIntegImg = False
        self.bb = None
        self.pLearner = None
        self.features = []
        self.kernels = []
        featureCounts = []
        for feat in self.config.features:
            self.features.append(HaarFeatures(self.config))
            featureCounts.append(self.features[-1].GetCount())
            kerType = feat.kernel
            if kerType == Config.kKernelTypeLinear:
                self.kernels.append(LinearKernel())
            elif kerType == Config.kKernelTypeGaussian:
                self.kernels.append(GaussianKernel())
            elif kerType == Config.kKernelTypeIntersection:
                self.kernels.append(IntersectionKernel())
            elif kerType == Config.kKernelTypeChi2:
                self.kernels.append(Chi2Kernel())
            # this should run for 1 iteration since we only use Haar Feature
        self.pLearner = LaRank(self.config, self.features[-1], self.kernels[-1])
        

    def Track(self, frame):
        #img = IntegImg(frame)
        img = ImageRep(frame)

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
        rects = Sampler.RadialSamples(self.bb, 2, self.config.searchRadius, 5, 16)
        keptRects = []
        keptRects.append(rects[0])
        for i, rect in enumerate(rects):
            if i < 1:
                continue
            if rect[i].IsInside(img.GetRect() :
                keptRects.append(rect)
        sample = MultiSample(img, keptRects)
        self.pLearner.Update(sample, 0)
