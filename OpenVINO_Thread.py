#! /usr/bin/python3

import numpy as np
import cv2
import datetime
from PIL import Image
from io import BytesIO
from imutils.video import FPS


## *** OpenVINO NCS/NCS2 (aka MYRIAD) AI Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
def AI_thread(results, inframe, net, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, dnnTarget, QUITf, blobThreshold, SSDv1):
    print("[INFO] OpenVINO " + dnnTarget + " AI thread" + str(tnum) + " is running...")
    fcnt=0
    waits=0
    drops=0
    prevDetections=list()
    for i in range(Ncameras):
        prevDetections.append(0)
    if tnum > 0:
        dnnTarget = dnnTarget + str(tnum)
    cfps = FPS().start()
    while not QUITf():     
        cameraLock.acquire()
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
        cameraLock.release()
        # get a frame
        try:
            (image, cam) = inframe[cq].get(True,0.100)
        except:
            image = None
            waits+=1
            continue
        if image is None:
            continue
        (h, w) = image.shape[:2]
        zoom=image.copy()   # for zoomed in verification run
        if SSDv1:
            blob = cv2.dnn.blobFromImage(cv2.resize(image, PREPROCESS_DIMS), 0.007843, PREPROCESS_DIMS, 127.5)
            personIdx=15
        else:
            blob = cv2.dnn.blobFromImage(image, size=PREPROCESS_DIMS)
            personIdx=1
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        imageDT = datetime.datetime.now()
        # loop over the detections, pretty much straight from the PyImageSearch sample code.
        personDetected = False
        ndetected=0
        fcnt+=1
        cfps.update()    # update the FPS counter
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        for i in np.arange(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]   # extract the confidence (i.e., probability) 
            idx = int(detections[0, 0, i, 1])   # extract the index of the class label
            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if conf > confidence and idx == personIdx and not np.array_equal(prevDetections[cam], detections):
                # then compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX=max(1, startX)
                startY=max(1, startY)
                endX=min(endX, w-1)
                endY=min(endY,h-1)
                xlen=endX-startX
                ylen=endY-startY
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                # adhoc "fix" for out of focus blobs close to the camera
                # out of focus blobs sometimes falsely detect -- insects walking on camera, etc.
                # In my real world use I have some static false detections, mostly under IR or mixed lighting -- hanging plants etc.
                # I put camera specific adhoc filters here based on (xlen,ylen,xcenter,ycenter)
                # TODO: come up with better way to do it, probably return (xlen,ylen,xcenter,ycenter) and filter at saving or Notify step.
                if float(xlen*ylen)/(w*h) > blobThreshold:     # detection filling too much of the frame is bogus
                   continue
                # display and label the prediction
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{}  {}".format(conf * 100,
                         str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), dnnTarget)
                cv2.putText(image, label, (2, (h-5)-(ndetected*28)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                personDetected = True
                initialConf=conf
                ndetected+=1
                break   # one is enough
        prevDetections[cam]=detections
        if personDetected and initialConf < noVerifyNeeded:  
            personDetected = False  # repeat on zoomed detection box
            try:
                # expand detection box by 10% for verification
                startY=int(0.9*startY)
                startX=int(0.9*startX)
                endY=min(int(1.1*endY),h-1)
                endX=min(int(1.1*endX),w-1)
                img = cv2.resize(zoom[startY:endY, startX:endX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
            except Exception as e:
                print(dnnTarget + " crop region ERROR: ", startY, endY, startX, endX)
                continue
            (h, w) = img.shape[:2]
            if SSDv1:
                blob = cv2.dnn.blobFromImage(img, 0.007843, PREPROCESS_DIMS, 127.5)
            else:
                blob = cv2.dnn.blobFromImage(img, size=PREPROCESS_DIMS)
            net.setInput(blob)
            detections = net.forward()
            imgDT = datetime.datetime.now()
            cfps.update()    # update the FPS counter
            tlabel = "{:.1f}%  ".format(initialConf * 100) + dnnTarget
            cv2.putText(img, tlabel, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            boxPointsV = (0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, 0, 0, 0, 0 only first four are used for dbg plots
            for i in np.arange(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                if  idx == personIdx:
                    if dbg:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        boxPointsV = (startX,startY, endX,endY, 0,0, 0,0)
                        label = "{:.1f}%  ".format(conf * 100) + dnnTarget +"V"
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(img, label, (2, (h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    if conf > verifyConf:
                        text = "Verify: {:.1f}%".format(conf * 100)   # show verification confidence 
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        personDetected = True
                        break
        else:
            ndetected = 0   # flag that no verification was needed
        # Queue results
        try:
            if personDetected:
                results.put((image, cam, personDetected, imageDT, dnnTarget, boxPoints), True, 1.0)    # try not to drop frames with detections
            else:
                if dbg is True and ndetected == 1:  # I want to see what the "zoom" has rejected
                    results.put((img, cam, True, imgDT, dnnTarget+"V", boxPointsV), True, 1.0) # force zoom rejection file write
                results.put((image, cam, personDetected, imageDT, dnnTarget, boxPoints), True, 0.016)
        except:
            # presumably outptut queue was full, main thread too slow.
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("OpenVINO " + dnnTarget + " AI thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))



