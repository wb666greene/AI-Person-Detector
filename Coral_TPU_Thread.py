#! /usr/bin/python3
### 16APR2021wbk
# Google has "obsoleted" the Python edgetpu API and replaced it with the PyCoral API
# Unfortunately the old and the new can't coexist on the same system.  The issue seems to be at a system library
# level and doesn't seem solvable with Python virtual environments.  Seems goofy to me that they couldn't just
# give the new incompatible system library a different name so they could coexist.
#
# I've added conditional code to allow the "legacy" edgetpu library to work if its there, and swtich to the PyCoral
# API if its not.  As far as I can tell, the main benefit to PyCoral is that it supports the new M.2 & MPCIe TPU hardware
# these may be a bit faster, but best is that they are < half the price of the USB3 TPU.  Now the search begins for inexpensive
# IOT class computers with M.2 and/or MPCIe slots.  I've tested the MPCIe module and this software on an i3-4025.
#

import numpy as np
import cv2
import datetime
from PIL import Image
from io import BytesIO
from imutils.video import FPS
# TPU
global __PYCORAL__
try:
    from edgetpu.detection.engine import DetectionEngine
    from edgetpu import __version__ as edgetpu_version
    __PYCORAL__ = False
except ImportError:
    print("[INFO]: Edgetpu support not installed, trying PyCoral")
    try:
        from pycoral.adapters import common, detect
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter, get_runtime_version
        __PYCORAL__ = True
        edgetpu_version=get_runtime_version()
    except ImportError:
        print("[ERROR]: Coral TPU support is not installed, exiting ...")
        quit()

print('Edgetpu_api version: ' + edgetpu_version)


## *** Coral TPU Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
def AI_thread(results, inframe, model, labels, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, QUITf, blobThreshold):
    global __PYCORAL__
    print("[INFO] Coral TPU AI thread" + str(tnum) + " is running...")
    waits=0
    drops=0
    fcnt=0
    if tnum > 0:
        ai = "TPU" + str(tnum)
    else:
        ai = "TPU"
    cfps = FPS().start()
    while not QUITf():
        cameraLock.acquire()
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
        cameraLock.release()
        # get a frame
        try:
            (image, cam, imageDT) = inframe[cq].get(True,0.100)
        except:
            image = None
            waits+=1
            continue
        if image is None:
            continue
        personDetected = False
        ndetected=0
        (h,w)=image.shape[:2]
        zoom=image.copy()   # for zoomed in verification run
        frame = cv2.cvtColor(cv2.resize(image, PREPROCESS_DIMS), cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # run the inference
        if __PYCORAL__ is False:
            if edgetpu_version < '2.11.2':
                detection = model.DetectWithImage(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
            else:
                detection = model.detect_with_image(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
        else:
            common.set_input(model,frame)
            model.invoke()
            detection=detect.get_objects(model, confidence, (1.0,1.0))
        cfps.update()    # update the FPS counter
        fcnt+=1
###        imageDT = datetime.datetime.now()
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        # loop over the detection results
        for r in detection:
            found=False
            if __PYCORAL__ is False:
                if r.label_id == 0:
                    # extract the bounding box and box and predicted class label
                    box = r.bounding_box.flatten().astype("int")
                    ##detect_label = labels[r.label_id] #not being used anywhere now
                    (startX, startY, endX, endY) = box.flatten().astype("int")
                    found=True
            else:
                if r.id == 0:
                    startX=r.bbox.xmin
                    startY=r.bbox.ymin
                    endX=r.bbox.xmax
                    endY=r.bbox.ymax
                    found=True
            if found:
                initialConf=r.score
                X_MULTIPLIER = float(w) / PREPROCESS_DIMS[0]
                Y_MULTIPLIER = float(h) / PREPROCESS_DIMS[1]
                startX = int(startX * X_MULTIPLIER)
                startY = int(startY * Y_MULTIPLIER)
                endX = int(endX * X_MULTIPLIER)
                endY = int(endY * Y_MULTIPLIER)
                xlen=endX-startX
                ylen=endY-startY
                if float(xlen*ylen)/(w*h) > blobThreshold:     # detection filling too much of the frame is bogus
                   continue
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                # draw the bounding box and label on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(initialConf * 100,
                        str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), ai)
                cv2.putText(image, label, (2, (h-5)-(ndetected*28)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                personDetected = True
                ndetected=ndetected+1
                break   # one person detection is enough
        # zoom in and repeat inference to verify detection
        if personDetected and initialConf < noVerifyNeeded:
            personDetected = False  # repeat on zoomed detection box
            try:
                ## removing this box expansion really hurt the verification sensitivity
                ## so try not expanding really small detections as my false positive was 89x144 so don't expand small boxes
                if max(xlen,ylen) > 150:
                    # expand detection box by 15% for verification
                    startY=int(0.85*startY)
                    startX=int(0.85*startX)
                    endY=min(int(1.15*endY),h-1)
                    endX=min(int(1.15*endX),w-1)
                else:
                    # expand by 5%
                    startY=int(0.95*startY)
                    startX=int(0.95*startX)
                    endY=min(int(1.05*endY),h-1)
                    endX=min(int(1.05*endX),w-1)
                img = cv2.resize(zoom[startY:endY, startX:endX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
            except Exception as e:
                print(" Coral crop region ERROR: {}:{} {}:{}", startY, endY, startX, endX)
                continue
            (h, w) = img.shape[:2]  # this will be PREPROCESS_DIMS (300, 300)
            if (h,w) != PREPROCESS_DIMS:
                print(" Bad resize!")
                continue
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            # run the zoom in and verify inference
            if __PYCORAL__ is False:
                if edgetpu_version < '2.11.2':
                    detection = model.DetectWithImage(frame, threshold=verifyConf, keep_aspect_ratio=True, relative_coord=False)
                else:
                 detection = model.detect_with_image(frame, threshold=verifyConf, keep_aspect_ratio=True, relative_coord=False)
            else:
                common.set_input(model,frame)
                model.invoke()
                detection=detect.get_objects(model, verifyConf, (1.0,1.0))
            cfps.update()    # update the FPS counter
            # loop over the detection results
            imgDT = datetime.datetime.now()
            tlabel = "{:.1f}%  ".format(initialConf * 100) + ai
            cv2.putText(img, tlabel, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            boxPointsV = (0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, 0, 0, 0, 0 only first four are used for dbg plots
            for r in detection:
                found=False
                if __PYCORAL__ is False:
                    if r.label_id == 0:
                        found=True
                        box = r.bounding_box.flatten().astype("int")
                        (startX, startY, endX, endY) = box.flatten().astype("int")
                else:
                    if r.id == 0:
                        found=True
                        startX=r.bbox.xmin
                        startY=r.bbox.ymin
                        endX=r.bbox.xmax
                        endY=r.bbox.ymax
                if found:
                    if dbg:
                        text = "{}: {:.1f}%".format("Person", initialConf * 100)
                        cv2.putText(img, text, (2, min(270,endY)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
                        # draw the person bounding box and label on the verification image and show verify confidence
                        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
                        text = "{:.1f}% CoralV".format(r.score * 100)
                        cv2.putText(img, text, (2, (h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    if r.score > verifyConf:
                        personDetected = True
                        text = "Verify: {:.1f}%".format(r.score * 100)   # show verification confidence on detection image
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        break
        else:
            ndetected=0     # flag that verification not needed
        # Queue results
        try:
            if personDetected:
                results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 1.0)    # try not to drop frames with detections
            else:
                if dbg is True and ndetected == 1:  # I want to see what the "zoom" has rejected
                    results.put((img, cam, True, imgDT, ai + "v", boxPointsV), True, 1.0) # force zoom rejection file write
                # change image to zoom if you don't want unverified detection box on displayed/saved image, at low frame rates the box is informative
                results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 0.016)
        except:
            # presumably outptut queue was full, main thread too slow.
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("Coral TPU thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))



