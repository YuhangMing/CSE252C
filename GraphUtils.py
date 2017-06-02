
# coding: utf-8

# In[1]:

import sys
import cv2
import numpy as np
from PIL import Image

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
LIGHTBLUE = (60,60,255)
LIGHTGREEN = (60,255,60)
LIGHTRED = (255,60,40)
BLUEGREEN = (0,210,210)
REDGREEN = (180,210,0)
REDBLUE = (210,0,180)
DARKBLUE = (0,0,185)
DARKGREEN = (0,185,0)
DARKRED = (185,0,0)
DARKGREY = (200,200,200)
customGraphColor = (0, 0, 0)
countGraph = 0
usingCustomGraphColor = False

# Get a color to draw graphs
def getGraphColor():
    if usingCustomGraphColor:
        usingCustomGraphColor = False
        return customGraphColor
    countGraph += 1
    if countGraph == 1:
        return LIGHTBLUE
    elif countGraph == 2:
        return LIGHTGREEN
    elif countGraph == 3:
        return LIGHTRED
    elif countGraph == 4:
        return BLUEGREEN
    elif countGraph == 5:
        return REDGREEN
    elif countGraph == 6:
        return REDBLUE
    elif countGraph == 7:
        return DARKBLUE
    elif countGraph == 8:    
        return DARKGREEN
    elif countGraph == 9:
        return DARKRED
    else:
        countGraph = 0
        return DARKGREY

#set the color used for graph
def setGraphColor(index = 0):
    countGraph, usingCustomGraphColor = index, False
    
#specify the color that will be used
def setCustomGraphColor(R, G, B):
    customGraphColor, usingCustomGraphColor = (R, G, B), True
    
def drawGraph(arraySrc, nArrayLength, imageDst, minV=0, maxV=0, width=0, height=0, graphLabel='', showScale=True):
    w, h, b = width, height, 10
    w = nArrayLength + b*2 if w <= 20 else w
    h = 220 if h <= 20 else h
    s, xscale = h-b*2, 1.0
    if nArrayLength > 1:
        xscale = (w-b*2) / float(nArrayLength-1)
    
    # Assume imageDst is a numpy array
    if not imageDst:
        imageGraph = Image.new('RGB', (w,h), WHITE)
        # imageGraph.show()
    else:
        imageGraph = imageDst
    if not imageGraph:
        print 'Error in drawGraph'
        return
    
    colorGraph = getGraphColor()
    if abs(minV) < 1e-7 and abs(maxV) < 1e-7:
        for i in range(nArrayLength):
            v = arraySrc[i]
            minV = v if v < minV else minV
            maxV = v if v > maxV else maxV
    
    diffV = maxV - minV
    diffV = 1e-7 if diffV == 0 else diffV
    fscale = float(s) / diffV
    
    # Draw the horizontal and vertical axis
    y0 = cv2.Round(minV*fscale)
    cv2.Line(imageGraph, (b, h-(b-y0)), (w-b, h-(b-y0)), BLACK)
    cv2.Line(imageGraph, (b, h-b), (b, h-(b+s)), BLACK)
    
    # Write the scale of the y and x axis
    if showScale:
        #cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        cv2.putText(imageGraph, '%.1f'%maxV, (1, b+4), cv2.FONT_HERSHEY_PLAIN, 2, GREY, 0.1)
        cv2.putText(imageGraph, '%d'%(nArrayLength-1), (w-b+4-5*2, h/2+10), cv2.FONT_HERSHEY_PLAIN, 2, GREY, 0.1)
        
    # Draw the values
    ptPrev = (b, h-(b-y))
    for i in range(nArrayLength):
        y = cv2.Round((arraySrc[i]-minV)*fscale)
        x = cv2.Round(i*xscale)
        ptNew = (b+x, h-(b+y))
        cv2.Line(imageGraph, ptPrev, ptNew, colorGraph, 1, cv2.CV_AA)
        ptPrev = ptNew
        
    # Write the graph label
    if graphLabel != None and graphLabel != '':
         cv2.putText(imageGraph, graphLabel, (30, 10), cv2.FONT_HERSHEY_PLAIN, 2, GREY, 0.1)
        
    return imageGraph

def showGraph(name, arraySrc, nArrayLength, delay_ms=500, background=None):
    imageGraph = drawGraph(arraySrc, nArrayLength, background)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, imageGraph)
    cv2.waitKey(10)
    cv2.waitKey(delay_ms)
    #cv2.destroyAllWindows()
    
            
    
    
    
    


# In[ ]:



