
# coding: utf-8

# In[ ]:

import sys
import math
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./Config.py')
from Config import Config
from copy import deepcopy

class Sampler:

    # @staticmethod
    # consider input "centre" is a Rect
    def __init__(self):
        pass

    def RadialSamples(self, centre, radius, nr, nt):
        # fRect = Rect(centre)
        fRect = Rect()
        fRect.initFromRect(centre)
        samples, rstep, tstep = [], float(radius)/nr, 2*float(math.pi)/nt
        samples.append(deepcopy(fRect))
        for ir in xrange(1, nr+1):
            phase = (ir%2)*tstep/2.0
            for it in xrange(nt):
                theta = it*tstep+phase
                dx, dy = ir*rstep*math.cos(theta), ir*rstep*math.sin(theta)
                fRect.SetXMin(centre.XMin() + dx)
                fRect.SetYMin(centre.YMin() + dy)
                samples.append(deepcopy(fRect))
        return samples
    
    # @staticmethod
    def PixelSamples(self, centre, radius, halfSample=False):
        # iRect = Rect(centre, isInt=True)    
        iRect = Rect()
        iRect.initFromRect(centre)
        
        samples = []
        samples.append(deepcopy(iRect))
        radius = int(radius)
        for iy in xrange(-radius, radius+1):
            for ix in xrange(-radius, radius+1):
                if ix*ix+iy*iy > radius*radius:
                    continue
                if iy is 0 and ix is 0:
                    continue
                if halfSample and ((ix%2 is not 0) or (iy%2 is not 0)):
                    continue
                iRect.SetXMin(int(centre.XMin()+ix))
                iRect.SetYMin(int(centre.YMin()+iy))
                samples.append(deepcopy(iRect))
        return samples
        

