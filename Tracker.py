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
        print("integral Image")
        print(img)
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
        s = Sampler()
        rects = s.PixelSamples(self.bb, self.config.searchRadius)
        print("sample size %f" % len(rects))
        keptRects = []
        for rect in rects:
            if( rect.isInside(img.GetRect()) ):
                keptRects.append(rect)

        print("kept %f rects" % len(keptRects)) 
        sample  = MultiSample(img, keptRects)
        scores = []
        self.pLearner.Eval(sample, scores)
        print("finished evaluation")	
        
        bestScore = max(scores)
        try:
            bestIndex = scores.index(bestScore)
        except ValueError:
            bestIndex = -1

        if not bestIndex == -1:
            self.bb = keptRects[bestIndex]
            self.UpdateLearner(img)

    def UpdateLearner(self, img):
        s = Sampler()
        rects = s.RadialSamples(self.bb, 2*self.config.searchRadius, 5, 16)
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
    
    
