
# coding: utf-8

# In[1]:

import sys
# import getopt
import random
import cv2
import linecache

import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./Config.py')
from Config import Config
sys.path.append('./Tracker.py')
from Tracker import Tracker

MAXINT = 100000
kLiveBoxWidth, kLiveBoxHeight = 80, 80
WHITE = (255, 255, 255)
GREEN = (60, 255, 60)


def rectangle(rMat, rRect, rColour):
    xmin, ymin, xmax, ymax = int(rRect.XMin()), int(
        rRect.YMin()), int(rRect.XMax()), int(rRect.YMax())
    cv2.rectangle(rMat, (xmin, ymin), (xmax, ymax), rColour)


def isFile(filePath, fileName):
    try:
        f = open(filePath, 'r')
    except IOError as err:
        print 'File error: ' + str(err) + fileName
        return False
    else:
        return True


def setCamera(conf):
    startFrame, endFrame, scaleW, scaleH, initBB = 0, MAXINT, 1.0, 1.0, Rect()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print 'Error: the video capture do not work.'
        return startFrame, endFrame, scaleW, scaleH, cap, initBB, False
    ret, frame = cap.read()
    if ret is True:
        scaleW, scaleH = float(conf.frameWidth) / frame.shape[1], float(conf.frameHeight) / frame.shape[0]
        initBB = Rect(int(conf.frameWidth / 2 - kLiveBoxWidth / 2), int(conf.frameHeight / 2 - kLiveBoxHeight / 2), int(kLiveBoxWidth), int(kLiveBoxHeight))
        print 'Press "i" to initialize tracker'
    else:
        print 'Error: video capture invalid.'
        return startFrame, endFrame, scaleW, scaleH, cap, initBB, False
    return startFrame, endFrame, scaleW, scaleH, cap, initBB, True

def getTrueRect(rectFilePath, lineid):
    line = linecache.getline(rectFilePath, lineid)
    words = filter(None, line.split())
    tRect = Rect(float(words[0]), float(words[1]), float(words[2]), float(words[3]))
    return tRect

def setFrame(conf):
    startFrame, endFrame, scaleW, scaleH, initBB = 0, MAXINT, 1.0, 1.0, Rect()

    frameFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/frame.txt'
    if isFile(frameFilePath, 'frame') is False:
        return startFrame, endFrame, scaleW, scaleH, initBB, False
    fframe = open(frameFilePath, 'r')
    line = fframe.readline()
    words = filter(None, line.split())
    if len(words) != 2 or int(words[0]) < 0 or int(words[0]) > int(words[1]):
        print 'Error: do not get the correct frame params.'    
        return startFrame, endFrame, scaleW, scaleH, initBB, False
    startFrame, endFrame = int(words[0]), int(words[1])
    fframe.close()

    # Get the info about the image
    imgFormatPath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/img/0001.jpg'
    if isFile(imgFormatPath, 'imgFormat') is False:
        return startFrame, endFrame, scaleW, scaleH, initBB, False
    img = cv2.imread(imgFormatPath, 0)
    scaleW = float(conf.frameWidth) / img.shape[1]
    scaleH = float(conf.frameHeight) / img.shape[0]

    # Get the groud truth rect
    rectFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/groundtruth_rect.txt'
    if isFile(rectFilePath, 'rect') is False:
        return startFrame, endFrame, scaleW, scaleH, initBB, False
    rRect = getTrueRect(rectFilePath, startFrame)
    initBB = Rect(rRect.XMin()*scaleW, rRect.YMin()*scaleH, rRect.Width()*scaleW, rRect.Height()*scaleH)
    return startFrame, endFrame, scaleW, scaleH, initBB, True



def main(argv=None):
    if argv is not None and len(argv) > 1:
        configPath = argv[1]
    else:
        configPath = './config.txt'
    conf = Config(configPath)

    if len(conf.features) == 0:
        print 'Error: no features specified in config'
        return 0

    resultsPath = './' + conf.resultsPath
    if isFile(resultsPath, 'res') is False:
        return 0
    fres = open(resultsPath, 'w')

    useCamera = (conf.sequenceName == '')

    if useCamera:
        startFrame, endFrame, scaleW, scaleH, cap, initBB, success = setCamera(conf)
    else:
        startFrame, endFrame, scaleW, scaleH, initBB, success = setFrame(conf)
    if not success:
        return 0

    tracker = Tracker(conf)
    if not conf.quietMode:
        cv2.namedWindow('Result')

    result = np.zeros((conf.frameHeight, conf.frameWidth, 3), np.uint8)
    paused, doInitialize = True, False
    random.seed(conf.seed)
    count, totalOverlap = 0, 0.0
    rectFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/groundtruth_rect.txt'

    for frameid in range(startFrame, endFrame + 1):
        if useCamera:
            ret, frameOrig = cap.read()
            if ret is True:
                frame = cv2.resize(frameOrig, (conf.frameWidth, conf.frameHeight))
                frame = cv2.flip(frame, 1)
                result = frame[:]
                if doInitialize:
                    if tracker.IsInitialized():
                        tracker.Reset()
                    else:
                        tracker.Initialize(frame, initBB)
                    doInitialize = False
                else:
                    if not tracker.IsInitialized():
                        rectangle(result, initBB, WHITE)
            else:
                print 'Error: video capture invalid.'
                return 0
        else:
            imgFramePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + ('/img/%04d.jpg' % int(frameid))
            if isFile(imgFramePath, 'imgFrame') is False:
                return False
            frameOrig = cv2.imread(imgFramePath, 0)
            if frameOrig is None:
                print 'Error: do not get valid input frame image.'
                return False
            frame = cv2.resize(frameOrig, (conf.frameWidth, conf.frameHeight))
            result = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            if frameid == startFrame:
                initBB.printStr()
                tracker.Initialize(frame, initBB)

        if tracker.IsInitialized():
            tracker.Track(frame)
            # if !conf.quietMode and conf.debugMode:
            #     tracker.Debug()
            
            box = tracker.GetBox()
            rectangle(result, box, GREEN)
            fRect = Rect(box.XMin() / scaleW, box.YMin() / scaleH, box.Width() / scaleW, box.Height() / scaleH)
            fres.write('%.2f %.2f %.2f %.2f' % (fRect.XMin(), fRect.YMin(), fRect.Width(), fRect.Height()))
            if not useCamera:
                tRect = getTrueRect(rectFilePath, frameid)
                overlap = float(tRect.Overlap(fRect))
                print 'true Rect: ',
                tRect.printStr()
                print 'tracker Rect: ',
                fRect.printStr()
                print 'overlap: %.2f' % overlap
                count, totalOverlap = count + 1, totalOverlap + overlap
                fres.write(' %.2f' % overlap)
            fres.write('\n')

        # print "hello image"

        if not conf.quietMode:
            # print 'show image'
            cv2.imshow("result", result)
            if paused:
                key = cv2.waitKey() & 0xFF
                print("manual")
                print("press i to start tracking in camera mode")
                print("press q to escape")
                print("press P to keep tracking")
                print("press any other key to track 1 frame")
            else:
                key = cv2.waitKey(1) & 0xFF
            if key != -1:
                if key == 27 or key == 113: # esc q
                    break
                elif key == 112: # p
                    paused = not paused
                elif key == 105 and useCamera: # i
                    doInitialize = True
                    print 'doInitialize'
            if conf.debugMode and frameid == endFrame:
                print 'End of sequence, press any key to exit.'
                cv2.waitKey()


    if useCamera:
        cap.release()
    else:
        score = totalOverlap / float(count)
        fres.write('%.3f\n' % score)
    if not fres.closed:
        fres.close()

    cv2.destroyAllWindows()
    return 1


if __name__ == "__main__":
    sys.exit(main())
