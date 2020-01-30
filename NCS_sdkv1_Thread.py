from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import datetime
from imutils.video import FPS



## *** NCS AI Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
def runNCS_AI(graph, image, PREPROCESS_DIMS):
    w=PREPROCESS_DIMS[0]
    h=PREPROCESS_DIMS[1]
    ncs_error=False
    # preprocess the image
    img = cv2.resize(image, PREPROCESS_DIMS)
    img = img - 127.5
    img = img * 0.007843
    img = img.astype(np.float16)
    # send the image to the NCS and run a forward pass to grab the network predictions
    try:
        predictions = []
        graph.LoadTensor(img, None)
        (output, _) = graph.GetResult()
    except Exception as e:
        ## So far I only see these errors on the Raspberry Pi and usually the Pi either needs to be rebooted,
        ## or the Movidius stick unplugged and plugged in again.  Hence the "Watchdog Timer" in the node-red "controller"
        ## I have some evidence that the "normal" 2.4A Pi power supply is not up to two NCS and marginal for one.
        currentDT = datetime.datetime.now()
        print("NCS Error: " + str(e)  + currentDT.strftime(" -- %Y-%m-%d %H:%M:%S.%f"))
        ncs_error = True
        return predictions, ncs_error
    # grab the number of valid object predictions from the output, then initialize the list of predictions
    num_valid_boxes = output[0]
    # loop over results, modifeid from PyImageSearch and other NCS sample code
    for box_index in range(int(num_valid_boxes)):
        # calculate the base index into our array so we can extract bounding box information
        base_index = 7 + box_index * 7

        # boxes with non-finite (inf, nan, etc) numbers must be ignored
        if (not np.isfinite(output[base_index]) or
            not np.isfinite(output[base_index + 1]) or
            not np.isfinite(output[base_index + 2]) or
            not np.isfinite(output[base_index + 3]) or
            not np.isfinite(output[base_index + 4]) or
            not np.isfinite(output[base_index + 5]) or
            not np.isfinite(output[base_index + 6])):
            continue
        # extract the image width and height and clip the boxes to the image size in case network returns boxes outside the image
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w, int(output[base_index + 5] * w))
        y2 = min(h, int(output[base_index + 6] * h))
        # grab the prediction class label, confidence (i.e., probability), and bounding box (x, y)-coordinates
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))
        # create prediciton tuple and append the prediction to the predictions list
        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)
    return predictions, ncs_error    # thread function for Movidius NCS AI
    
    
def AI_thread(results, inframe, graph, tnum, cameraLock, nextCamera, Ncameras,
                PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, QUITf, blobThreshold):
        NCSerror=False
        print("[INFO] NCS SDK V1 AI thread" + str(tnum) + " is running...")
        waits=0  # count of times thread input inframe queue was empty, no data for thread to process
        drops=0  # count of times frame dropped because main thread results queue was full
        fcnt=0
        if tnum > 0:
            ai = "NCS" + str(tnum)
        else:
            ai = "NCS"
        cfps = FPS().start()
        while not QUITf():
            cameraLock.acquire()
            cq=nextCamera
            nextCamera = (nextCamera+1)%Ncameras
            cameraLock.release()
            # get a frame
            try:
                (image, cam) = inframe[cq].get(True,0.010)
            except:
                image = None
                waits+=1
                continue
            if image is None:   # rare but seems to stopped a "crash"
                continue
            (hh,ww)=image.shape[:2]
            zoom=image.copy()   # copy to zoom in on detection for verification
            predictions,NCSerror = runNCS_AI(graph, image, PREPROCESS_DIMS)
            if NCSerror:
                break
            cfps.update()    # update the FPS counter
            fcnt+=1
            imageDT = datetime.datetime.now()
            # loop over our predictions
            personDetected=False
            ndetected=0
            boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
            for (i, pred) in enumerate(predictions):
                # extract prediction data for readability
                (pred_class, pred_conf, pred_boxpts) = pred
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if pred_conf > confidence and pred_class == 15:
                    (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                    # Support per camera frame sizes
                    X_MULTIPLIER = float(ww) / PREPROCESS_DIMS[0]
                    Y_MULTIPLIER = float(hh) / PREPROCESS_DIMS[1]
                    startX = int(ptA[0] * X_MULTIPLIER)
                    startY = int(ptA[1] * Y_MULTIPLIER)
                    endX = int(ptB[0] * X_MULTIPLIER)
                    endY = int(ptB[1] * Y_MULTIPLIER)
                    # adhoc "fix" for out of focus blobs close to the camera
                    xlen=endX-startX
                    ylen=endY-startY
                    xcen=int((startX+endX)/2)
                    ycen=int((startY+endY)/2)
                    boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                    # out of focus blobs sometimes falsely detect -- insects walking on camera, etc.
                    # In my real world use I have some static false detections, mostly under IR or mixed lighting -- hanging plants etc.
                    # I put camera specific adhoc filters here based on (xlen,ylen,xcenter,ycenter)
                    # TODO: come up with better way to do it, probably return (xlen,ylen,xcenter,ycenter) and filter at saving or Notify step.
                    if float(xlen*ylen)/(ww*hh) > blobThreshold:     # detection box filling too much of the frame is bogus
                        continue
                    personDetected=True
                    initialConf=pred_conf
                    # print prediction to terminal
                    label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {}".format(pred_conf * 100, 
                            str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), ai)
                    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
                    cv2.putText(image, label, (2, (hh-5)-(ndetected*28)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                    ndetected=ndetected+1
                    break   # no need for more detections one is sufficient
            if personDetected and initialConf < noVerifyNeeded:   # multiple detections have so far always had one valid person detection
                # filter bogus detections by zooming on detection and repeating inference
                try:
                    # expand detection box by 10% for verification
                    ##startY=int(0.9*startY)
                    ##startX=int(0.9*startX)
                    ##endY=min(int(1.1*endY),hh-1)
                    ##endX=min(int(1.1*endX),ww-1)
                    img = cv2.resize(zoom[startY:endY, startX:endX], PREPROCESS_DIMS, interpolation = cv2.INTER_AREA)
                except Exception as e:
                    print(" NCS crop region ERROR: ", startY, endY, startX, endX)
                    continue
                verify, NCSerror = runNCS_AI(graph, img, PREPROCESS_DIMS)
                if NCSerror:
                    break
                cfps.update()    # update the FPS counter
                imgDT = datetime.datetime.now()
                personDetected=False
                tlabel = "{:.1f}%  ".format(initialConf * 100) + ai
                cv2.putText(img, tlabel, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                boxPointsV = (0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, 0, 0, 0, 0 only first four are used for dbg plots
                for (i, pred) in enumerate(verify):
                    (pred_class, pred_conf, pred_boxpts) = pred
                    if pred_class == 15:
                        if dbg:
                            (hh,ww)=img.shape[:2]
                            (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                            startX = int(ptA[0])
                            startY = int(ptA[1])
                            endX = int(ptB[0])
                            endY = int(ptB[1])
                            boxPointsV = (startX,startY, endX,endY, 0,0, 0,0)
                            label = "{:.1f}% NCSv".format(pred_conf * 100)
                            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 1)
                            cv2.putText(img, label, (2, (hh-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                        if pred_conf > verifyConf:
                            text = "Verify: {:.1f}%".format(pred_conf * 100)   # show verification confidence 
                            cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            personDetected=True
                            break;
            else:
                ndetected = 0   # flag verifivation wasn't run
            # Queue results
            try:
                if personDetected:
                    results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 1.0)    # try not to drop frames with detections
                else:
                    if dbg is True and ndetected == 1:  # I want to see what the "zoom" rejected
                        results.put((img, cam, True, imgDT, ai + "v", boxPointsV), True, 1.0) # force zoom rejection file write
                    results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 0.016)
            except:
                # presumably outptut queue was full, main thread too slow.
                drops+=1
                continue
        # Thread exits
        cfps.stop()    # stop the FPS counter timer
        if NCSerror:
###            print("NCS SDK V1 AI thread ERROR EXIT!" + str(tnum) + " waited for input " + str(waits) + " times, dropped " + str(drops) + " output out of " + str(fcnt))
            print("NCS SDK V1 AI ERROR EXIT! thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
                + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))
        else:
###            print("NCS SDK V1 AI thread" + str(tnum) + " waited for input " + str(waits) + " times, dropped " + str(drops) + " output out of " + str(fcnt))
            print("NCS SDK V1 AI thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
                + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))



