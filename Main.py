import sys
import random
import cv2
import linecache
import re
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


def trackCamera(conf):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print 'Error: the video capture do not work.'
        return False

    ret, frame = cap.read()
    if ret is True:
        scaleW, scaleH = float(conf.frameWidth) / frame.shape[1], float(conf.frameHeight) / frame.shape[0]
        initBB = Rect(int(conf.frameWidth / 2 - kLiveBoxWidth / 2), int(conf.frameHeight / 2 - kLiveBoxHeight / 2), int(kLiveBoxWidth), int(kLiveBoxHeight))
        print 'Press "i" to initialize tracker'
    else:
        print 'Error: video capture invalid.'
        return False

    if not conf.quietMode:
        cv2.namedWindow('Result')

    tracker = Tracker(conf)
    startFrame, endFrame, paused, doInitialize, count = 0, MAXINT, True, False, 0
    result = np.zeros((conf.frameHeight, conf.frameWidth, 3), np.uint8)
    random.seed(conf.seed)

    for frameid in xrange(startFrame, endFrame + 1):
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
            return False

        if tracker.IsInitialized():
            tracker.Track(frame)
            # if !conf.quietMode and conf.debugMode:
            #     tracker.Debug()
            box = tracker.GetBox()
            rectangle(result, box, GREEN)

        # print "hello image"

        if not conf.quietMode:
            # print 'show image'
            cv2.imshow("result", result)
            if paused:
                key = cv2.waitKey() & 0xFF
                print("manual")
                print("press i to start tracking in camera mode")
                print("press q or esc to escape")
                print("press p to keep tracking")
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

    cap.release()
    cv2.destroyAllWindows()
    return True


def getTrueRect(rectFilePath, lineid):
    line = linecache.getline(rectFilePath, lineid)
    words = filter(None, re.split('; |,', line))
    tRect = Rect(float(words[0]), float(words[1]), float(words[2]), float(words[3]))
    return tRect


def trackFrame(conf):
    frameFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/frame.txt'
    if isFile(frameFilePath, 'frame') is False:
        return False
    fframe = open(frameFilePath, 'r')
    line = fframe.readline()
    words = filter(None, re.split('; |,', line))
    if len(words) != 2 or int(words[0]) < 0 or int(words[0]) > int(words[1]):
        print 'Error: do not get the correct frame params.'    
        return False
    startFrame, endFrame = int(words[0]), int(words[1])
    fframe.close()

    # Get the info about the image
    imgFormatPath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/img/0001.jpg'
    if isFile(imgFormatPath, 'imgFormat') is False:
        return False
    img = cv2.imread(imgFormatPath, 0)
    scaleW = float(conf.frameWidth) / img.shape[1]
    scaleH = float(conf.frameHeight) / img.shape[0]

    # Get the groud truth rect
    rectFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/groundtruth_rect.txt'
    if isFile(rectFilePath, 'rect') is False:
        return False
    rRect = getTrueRect(rectFilePath, startFrame)
    initBB = Rect(rRect.XMin()*scaleW, rRect.YMin()*scaleH, rRect.Width()*scaleW, rRect.Height()*scaleH)


    if not conf.quietMode:
        cv2.namedWindow('Result')

    tracker = Tracker(conf)
    paused, doInitialize, count, totalOverlap = True, False, 0, 0.0
    random.seed(conf.seed)
    result = np.zeros((conf.frameHeight, conf.frameWidth, 3), np.uint8)
    rectFilePath = './' + conf.sequenceBasePath + '/' + conf.sequenceName + '/groundtruth_rect.txt'

    for frameid in xrange(startFrame, endFrame + 1):
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

            tRect = getTrueRect(rectFilePath, frameid)
            overlap = float(tRect.Overlap(fRect))
            print 'true Rect: ',
            tRect.printStr()
            print 'tracker Rect: ',
            fRect.printStr()
            print 'overlap: %.2f' % overlap
            count, totalOverlap = count + 1, totalOverlap + overlap
            fres.write(' %.2f\n' % overlap)

        # print "hello image"
        if not conf.quietMode:
            # print 'show image'
            cv2.imshow("result", result)
            if paused:
                key = cv2.waitKey() & 0xFF
                print("manual")
                print("press i to start tracking in camera mode")
                print("press q or esc to escape")
                print("press p to keep tracking")
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

    score = totalOverlap / float(count)
    fres.write('%.3f\n' % score)
    if not fres.closed:
        fres.close()

    cv2.destroyAllWindows()
    return True



def main(argv=None):
    if argv is not None and len(argv) > 1:
        configPath = argv[1]
    else:
        configPath = './config.txt'
    conf = Config(configPath)

    if len(conf.features) == 0:
        print 'Error: no features specified in config'
        return False

    resultsPath = './' + conf.resultsPath
    if isFile(resultsPath, 'res') is False:
        return False
    fres = open(resultsPath, 'w')

    useCamera = (conf.sequenceName == '')

    if useCamera:
        return trackCamera(conf)
    else:
        return trackFrame(conf)


if __name__ == "__main__":
    sys.exit(main())
