
# coding: utf-8

# In[1]:

import sys
import getopt
import cv2
import numpy as np
sys.path.append('./Rect.py')
from Rect import Rect
sys.path.append('./Config.py')
from Config import Config
sys.path.append('./Tracker.py')
from Tracker import Tracker
import random

kLiveBoxWidth, kLiveBoxHeight = 80, 80
WHITE = (255, 255, 255)
GREEN = (60,255,60)

def rectangle(rMat, rRect, rColour):
    xmin, ymin, xmax, ymax = int(rRect.XMin()), int(rRect.YMin()), int(rRect.XMax()), int(rRect.YMax())
    cv2.rectangle(rMat, (xmin, ymin ), (xmax, ymax), rColour)

def isFile(filePath, fileName):
    if os.path.isfile(filePath) == False:
        print 'Error: could not load %s file %s' %(fileName, filePath)
        return False
    else:
        return True

def main(argv=None):
    configPath = argv[1] if len(argv) > 1 else '../config.txt'
    conf = Config(configPath)
    # conf.PrintStr()
    
    if len(conf.features) == 0:
        print 'Error: no features specified in config'
        return 0

    if isFile(conf.resultsPath, 'conf') == False:
        return 0
    fres = open(conf.resultsPath,'w')
    
    useCamera = (conf.sequenceName=='')
    startFrame, endFrame, scaleW, scaleH = -1, -1, 1.0, 1.0
    
    if useCamera:
        # Do the camera traning
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print 'Error: the video capture do not work.'
            return 0
        startFrame, endFrame = 0, INT_MAX
        ret, frame = cap.read()
        if ret == True:
            scaleW, scaleH = float(conf.frameWidth)/frame.shape[1], float(conf.frameHeight)/frame.shape[0]
            iRect = Rect(int(conf.frameWidth/2-kLiveBoxWidth/2), int(conf.frameHeight/2-kLiveBoxHeight/2),                         int(kLiveBoxWidth), int(kLiveBoxHeight))
            initBB = initFromRect(iRect)
            print 'Press "i" to initialize tracker'
        else:
            print 'Error: video capture invalid.'
            return 0
    else:
        # Do the sequence training.
        # Get the start and end frame
        frameFilePath = conf.sequenceBasePath + '/' + conf.sequenceName + '/frame.txt'
        if isFile(frameFilePath, 'frame') == False:
            return 0
        fframe = open(frameFilePath,'r')
        line = frect.readline()
        words = filter(None, line.split())
        if len(words) != 2 or int(words[0]) < 0 or int(words[0]) > int(words[1]):
            print 'Error: do not get the correct frame params.'
            return 0
        startFrame, endFrame = int(words[0]), int(words[1])

        # Get the info about the image
        imgFormatPath = conf.sequenceBasePath + '/' + conf.sequenceName + '/img/0001.jpg'
        if isFile(imgFormatPath, 'imgFormat') == False:
            return 0
        img = cv2.imread(imgFormatPath, 0)
        scaleW = float(conf.frameWidth) / img.shape[1]
        scaleH = float(conf.frameHeight) / img.shape[0]

        # Get the groud truth rect
        rectFilePath = conf.sequenceBasePath + '/' + conf.sequenceName + '/groundtruth_rect.txt'
        if isFile(rectFilePath, 'rect') == False:
            return 0
        frect = open(rectFilePath,'r')
        frect.getline()
        words = filter(None, line.split())
        if len(words) != 4:
            print 'Error: do not get the correct frame params.'
            return 0
        xmin, ymin, width, height = float(words[0]), float(words[1]), float(words[2]), float(words[3])
        fRect = Rect(xmin*scaleW, ymin*scaleH, width*scaleW, height*scaleH)
        initBB = initFromRect(fRect)
            
            
    tracker = Tracker(conf)
    if not conf.quietMode:
        cv2.namedWindow('Result')


    #result = cv.createMat(conf.frameHeight, conf.frameWidth, cv.CV_8UC3)
    result = np.zeros((conf.frameHeight, conf.frameWidth, 3), np.uint8)
    paused, doInitialize = False, False
    random.seed(conf.seed)
    for frameid in range(startFrame, endFrame+1):
        if useCamera:
            ret, frameOrig = cap.read()
            if ret==True:
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
                print 'Error: do not capture valid video.'
        else:

            imgFramePath = conf.sequenceBasePath + '/' + conf.sequenceName + ('/img/%04d.jpg' % int(frameid))
            if isFile(imgFramePath, 'imgFrame') == False:
                return 0
            frameOrig = cv2.imread(imgFramePath, 0)
            if frameOrig == None:
                print 'Error: do not get valid input frame image.'
                return 0
            frame = cv2.resize(frameOrig, (conf.frameWidth, conf.frameHeight))
            result = cv2.cvtColor(frame, cv.COLOR_RGB2GRAY)

            if frameid == startFrame:
                tracker.Initialize(frame, initBB)

        if tracker.IsInitialized():
            tracker.Track(frame)

            # incomplete tracker
            # if !conf.quietMode and conf.debugMode:
            #     tracker.Debug()

            rectangle(result, tracker.GetBox(), GREEN)

            box = Tracker.GetBox()
            fres.write('%d %d %d %d\n' %(box.XMin()/scaleW, box.YMin()/scaleH, box.Width()/scaleW, box.Height()/scaleH))

        if not conf.quietMode:
            cv2.imshow("result", result)
            key = cv2.waitKey(0 if paused else 1)
            if key != -1:
                if key==27 or key==113:
                    break 
                elif key==112:
                    paused = not paused
                elif key==105 and useCamera:
                    doInitialize = True
            if conf.debugMode and frameid==endFrame:
                print 'End of sequence, press any key to exit.'
                cv2.waitKey()

    cv2.destroyAllWindows()
    cap.release()
    return 1


if __name__ == "__main__":
    sys.exit(main())
    

