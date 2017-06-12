#!/usr/bin/env python
import numpy as np
import cv2

from Rect import Rect
from Config import Config
from Tracker import Tracker
# from ImageRep import ImageRep
# from Sampler import Sampler
# from Sample import Sample
# from Sample import MultiSample
# from HaarFeatures import HaarFeatures
# import Kernels
# from LaRank import LaRank

import random
import linecache


def getTrueRect(rectFilePath, lineid):
    line = linecache.getline(rectFilePath, lineid)
    words = filter(None, line.split())
    
    tRect = Rect(float(words[0]), float(words[1]), float(words[2]), float(words[3]))
    return tRect


if __name__ == "__main__":
    # c = Config("./config.txt")
    # t = Tracker(c)

    # cv2.namedWindow("preview")
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

    ## Yohann
    ##########################
    ## modify cofig.txt file
    ## modify frame.txt file
    ##########################

    c = Config("./config.txt")
    t = Tracker(c)
    cv2.namedWindow("preview")

    # get image info
    imgFormatPath = './' + c.sequenceBasePath + '/' + c.sequenceName + '/img/0001.jpg'
    tmp = cv2.imread(imgFormatPath, 0)
    scaleW = float(c.frameWidth) / tmp.shape[1]
    scaleH = float(c.frameHeight) / tmp.shape[0]
    
    # frame file bbox
    frameFilePath = './' + c.sequenceBasePath + '/' + c.sequenceName + '/frame.txt'
    try:
        fframe = open(frameFilePath, 'r')
    except IOError as err:
        print(" rectangle path error")
    # fframe = open(frameFilePath, 'r')
    line = fframe.readline()
    words = filter(None, line.split())
    startFrame, endFrame = int(words[0]), int(words[1])
    fframe.close()

    # ground truth bbox file
    rectFilePath = './' + c.sequenceBasePath + '/' + c.sequenceName + '/groundtruth_rect.txt'
    try:
        frect = open(rectFilePath, 'r')
    except IOError as err:
        print(" rectangle path error")
    # frect = open(rectFilePath, 'r')
    line = frect.readline()
    # get initial box
    words = filter(None, line.split())
    xmin, ymin, width, height = float(words[0]), float(words[1]), float(words[2]), float(words[3])
    initBB = Rect(int(xmin * scaleW), int(ymin * scaleH), int(width * scaleW), int(height * scaleH))
    frect.close()
    # initBB.printStr()

    # open result path
    # if isFile(c.resultsPath, 'res') is False:
    #     print(" results path wrong")
    try:
        fres = open(c.resultsPath, 'w')
    except IOError as err:
        print(" results path error")
        # return False
    # else:
    #     print(" results file opened")
    #     fres = open(c.resultsPath, 'w')
    # fres = open(c.resultsPath, 'w')

    overlapAll = 0.0
    ovarlapAllBin = 0.0
    for i in range(1, endFrame+1):
        frame = cv2.imread('./data/'+ c.sequenceName +'/img/%04d.jpg' % i, 0)
        print("frame #", i)
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

        fRect = t.GetBox()
        rectSB = Rect(int(fRect.XMin()/scaleW), int(fRect.YMin()/scaleH), int(fRect.Width()/scaleW), int(fRect.Height()/scaleH))
        # rectSB.printStr()
        
        # get GT bbox
        tRect = getTrueRect(rectFilePath, i)
        # tRect.printStr()

        overlap = float(tRect.Overlap(rectSB))
        if (overlap>=0.5):
            overlapBin = 1.0
        else:
            overlapBin = 0.0
        print(overlap)
        overlapAll += overlap
        ovarlapAllBin += overlapBin

        # print(int(fRect.XMin()/scaleW), int(fRect.YMin()/scaleH), int(fRect.Width()/scaleW), int(fRect.Height()/scaleH))
        fres.write('%d %d %d %d %f %f' % (rectSB.XMin(), rectSB.YMin(), rectSB.Width(), rectSB.Height(), overlap, overlapBin))
        fres.write("\n")
        # uBB = t.GetBox()
        # print(uBB.XMin(), uBB.YMin(), uBB.Width(), uBB.Height())
        # t.Initialize(frame, initBB)
        key = cv2.waitKey(20)

    overlapAvg = overlapAll / endFrame
    overlapAvgBin = ovarlapAllBin / endFrame
    fres.write('average overlap score: %f' % overlapAvg)
    fres.write("\n")
    fres.write('average overlap binary score: %f' % overlapAvgBin)
    fres.close()
    cv2.destroyWindow("preview")
    
    
