import numpy as np
import cv2
import requests
import time
import datetime
from PIL import Image
from io import BytesIO


## *** ONVIF Sampling Thread ***
#******************************************************************************************************************
# Onvif camera sampling thread
def onvif_thread(inframe, camn, URL, QUITf):
    print("[INFO] ONVIF Camera" + str(camn) + " thread is running...")
    ocnt=0  # count of times inframe thread output queue was full (dropped frames)
    Error=False
    while not QUITf():
        # grab the frame
        try:
            r = requests.get(URL)
            i = Image.open(BytesIO(r.content))
            frame = np.array(i)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if Error and frame is not None:   # log when it recovers
                currentDT = datetime.datetime.now()
                print('[******] Onvif cam'+ str(camn) + ' error has recovered: ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                Error=False    # after getting a good frame, enable logging of next error
        except Exception as e:
            # this appears to fix the Besder camera problem where it drops out for minutes every 5-12 hours
            if not Error:   # suppress the zillions of sequential error messages while it recovers
                currentDT = datetime.datetime.now()
                ## printing the error string hasn't been particularly informative
                ##print('Onvif cam'+ str(camn) + ': ' + str(e) + URL[0:30] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                print('[Error!] Onvif cam'+ str(camn) + ': ' +  URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            frame = None
            Error=True
            time.sleep(5.0)     # let other threads have more time while this camera recovers, which sometimes takes minutes
        try:
            if frame is not None and not QUITf():
                inframe.put((frame, camn), True, 0.200)
                ##time.sleep(sleepyTime)   # force thread switch, hopefully smoother sampling, 10Hz seems upper limit for snapshots
        except: # most likely queue is full
            if QUITf():
                break
            ocnt=ocnt+1
            ##time.sleep(sleepyTime)
            continue
    print("ONVIF Camera" + str(camn) + " thread is exiting, dropped frames " + str(ocnt) + " times.")


