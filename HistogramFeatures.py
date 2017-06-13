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

kNumBins, kNumLevels, kNumCellsX, kNumCellsY = 16, 4, 3, 3

class HistogramFeatures(Features):
    ## self.featList -> self.featList
    ## self.
    def __init__(self, conf):
        Features.__init__(self)
        nc = 0
        for i in range(kNumLevels):
            nc += (i+1)**2
        self.SetCount(kNumBins*nc)
        print "Histogram bins:", self.GetCount()

    def UpdateFeature(self, sam):
        rect = sam.GetROI()
        self.featList = [0 for x in self.featList]
        histind = 0
        for i in range(kNumLevels):
            nc= i + 1
            w, h = float(sam.GetROI().getWidth()/nc), float(sam.GetROI().getHeight()/nc)
            cell = Rect(0.0, 0.0, w, h)
            for iy in range(nc):
                cell.SetY(sam.GetROI().getY()+iy*h)
                for ix in range(nc):
                    cell.SetXMin(sam.GetROI().getX()+ix*w)
                    hist = sam.GetSelf().image.Hist(cell)
                    self.featList[histind*kNumBins:(histind+1)*kNumBins] = hist[:]
                    histind+=1
        # tmp_arr = np.array(self.featList)
        # tmp_arr/= histind
        # self.featList = tmp_arr.tolist()
        self.featList = [x/histind for x in self.featList]
        # self.featList /= histind




