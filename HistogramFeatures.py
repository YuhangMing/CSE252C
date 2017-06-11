import sys
import cv2 as cv
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
    def __init__(self, conf):
        Features.__init__(self)
        nc = 0
        for i in range(kNumLevels):
            nc += (i+1)**2
        self.SetCount(kNumBins*nc)
        print "Histogram bins:", self.GetCount()

    def UpdateFeature(self, sam):
        rect = sam.GetROI()
        self.m_featVec = [0 for x in self.m_featVec]
        histind = 0
        for i in range(kNumLevels):
            nc= i + 1
            w, h = float(sam.GetROI().Width()/nc), float(sam.GetROI().Height()/nc)
            cell = Rect(0.0, 0.0, w, h)
            for iy in range(nc):
                cell.SetYMin(sam.GetROI().YMin()+iy*h)
                for ix in range(nc):
                    cell.SetXMin(sam.GetROI().XMin()+ix*w)
                    hist = sam.GetSelf().m_image.Hist(cell)
                    self.m_featVec[histind*kNumBins:(histind+1)*kNumBins] = hist[:]
                    histind+=1
        self.m_featVec /= histind




