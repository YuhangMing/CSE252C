
# coding: utf-8

# In[ ]:

import sys
import math
import cv2 as cv
import numpy as np
sys.path.append('./Rect.py')
import Rect
sys.path.append('./Config.py')
import Config

class Sampler:

    @staticmethod

    def RadialSamples(centre, radius, nr, nt):
        # fRect = Rect(centre)
        self.fRect = Rect()
        self.fRect.initFromList(fRect)
        samples, rstep, tstep = [], float(radius)/nr, 2*float(math.pi)/nt
        samples.append(centre)
        for ir in range(1, nr+1):
            phase = (ir%2)*tstep/2.0
            for it in range(nt):
                theta = it*tstep+phase
                dx, dy = ir*rstep*math.cos(theta), ir*rstep*math.sin(theta)
                fRect.SetXMin(centre.XMin() + dx)
                fRect.SetYMin(centre.YMin() + dy)
                samples.append(fRect)
        return samples
    
    def PixelSamples(centre, radius, halfSample=False):
        iRect = Rect(centre, isInt=True)
        samples = []
        samples.append(iRect)
        radius = int(radius)
        for iy in range(-radius, radius+1):
            for ix in range(-radius, radius+1):
                if ix*ix+iy*iy > radius*radius:
                    continue
                if iy is 0 and ix is 0:
                    continue
                if halfSample and ((ix%2 is not 0) or (iy%2 is not 0)):
                    continue
                iRect.SetXMin(int(centre.XMin()+ix))
                iRect.SetYMin(int(centre.YMin()+iy))
                samples.append(iRect)
        return samples
        

