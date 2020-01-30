#!/usr/bin/env python3
#
# AI_OVmt.py 13APR2019wbk
#
#   This is in "maintainence" mode, only updates will be what is necessary keep working with AI_dev.py_Controler-Viewer.json.
#   I maintain it becuase its a bit more efficient on IOT hardware (Pi4, XU-4, etc.) and I think the OpenVINO MobileNet-SSD model
# is better in terms of false detections than the one "compiled" for NCS v1 SDK.  But MobileNet-SSDv2 on the TPU is even better!
# TODO: Get MobileNet-SSDv2 for OpenVINO.
#   It also has the Windows CPU only AI function, but its missed a lot of testing cycles so don't be surprised if its broken, as
# the need for running on Windows is basically gone.
#
# With 9 MQTT cameras via i5ai rtsp2mqttPdemant server and localhost AI controller saving detections, i7-6700K Desktop gets:
# ~11.9 fps with a single NCS stick, ~25.0 fps with NCS2 stick, and ~22.9 fps with 2 NCS sticks. 
#
#
##   derived from: AI_mt.py
## Quick hack to use OpenVINO instead of NCS v1 SDK for Movidius support.
## Using info from: https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi
##
## Some AI_OVmt.py OpenVINO test results:
## Seems a bit slower on the Pi3B+ with 5 Onvif cams giving ~5.9 fps vs ~6.5 fps for the v1 SDK.
## 17APR2019wbk,  Using 5 Onvif snaphot netcams:
## Pi3B+:
##      NCS v1 SDK ~6.5 fps
##      2 NCS v1 SDK ~11.6 fps
##      NCS OpenVINO ~5.9 fps
##      2 NCS OpenVINO ~9.9 fps
##      NCS2 OpenVINO ~8.3 fps
##
## Odroid XU-4:
##      NCS OpenVINO ~8.5 fps
##      2 NCS OpenVINO ~15.9 fps
##      NCS2 OpenVINO ~15.5 fps## Mixing NCS2 and NCS gets ~8.9 fps barely better than the NCS alone and worse than using 2 NCS.  Something weird here.
##
## i5 4200U:
##      1 NCS 0 CPU ~10.4 fps
##      1 NCS 1 CPU ~21.3 fps
##      2 NCS 0 CPU ~19.7 fps
##      2 NCS 1 CPU ~19.5 fps
##      1 NCS2 0 CPU ~22.5 fps
##      1 NCS2 1 CPU ~20.1 fps
##      0 NCS 2 CPU ~21.7 fps
##
##  i5 4200U 9 rtsp streams:
##      1 NCS2 0 CPU ~14.5 fps
##      1 NCS2 1 CPU ~7.7 fps       # rtsp has significant overhead.
##
##
## This should be the last version derived from AI_unified_mt.py since I plan to move the code from NCS v1 SDK to OpenVINO or Coral TPU
## Changes: Completed 18MAR2019wbk
##  1) Send images as MQTT buffer instead of local file system path, allows notifications to be on different machine
##  2) Add option to save all analized images to local storage, primarily for testing/debugging
##  3) Add round-robin sampling for rtsp cameras, could be useful for Lorex DVR where all are on the same server (minimally tested)
##     so-far, round-robin sampling seems inferior in my testing, probably shouldn't be used.
##  4) Allow mix of input methods -- rtsp, onvif, images via MQTT
##  Completed 20MAR2019wbk:
##  5) Add MQTT message buffer "front-end" to get images to analyze, typically from node-red ftp server node.
##  topic: MQTTcam/N, payload: buffer containing jpeg image
##
##
## 24MAR2019wbk
## Some AI_mt.py NCS v1 SDK test resluts:
## Note there is no NCS v1 API support on Windows.  (Intel indroduced this with OpenVINO, R5?)
## I don't run Windows, but I've done limited testing on a couple of friend's machines I set up for dual boot.
## i3 4025U:    Windows7    -- 5 onvif cameras gets ~6.2 fps, ~6.1 fps Windows10
##                          -- 5 rtsp streams gets ~5.2fps, ~5.3 fps Windows10
##              Ubuntu16.04 -- same hardware, CPU AI only, No NCS installed to compare with Windows (dual boot)
##                          -- 5 Onvif cameras gets ~8.4 fps
##                          -- 5 rtsp ~7.4 fps.
##
## Pi3B+:       CPU AI is too slow to be useful ~0.6 fps,  start program with option:  - nt 0
##              Python 2.7  -- 5 onvif cameras 1 NCS gets ~6.5 fps, 2 NCS doesn't run reliably giving NCS timeout errors, might be power suppy issue.  
##              Python 3.5  -- 5 onvif cameras 1 NCS gets ~6.6 fps, same erratic results with two NCS sticks, but short runs do ~11.7 fps
##                          -- 5 rtsp streams gets ~4.5 fps, 2 NCS gets ~6.2 fps  I don't think the Pi is a good choice for rtsp streams.           
##
## i5 M540:     Ubuntu16.04 -- 5 onvif, CPU AI with old CPU without AVX, AVX2, does poorly ~3.9 fps, better than Pi3 but worse than i3 4025U
##                          -- 5 onvif, 1 NCS, no CPU AI gets ~8.8 fps, 2 NCS ~17.0 fps, 1 NCS + 1 CPU AI ~12.0 fps, lack of USB3 holds it back
##                          -- 5 rtsp, 1 NCS, no CPU AI ~8.8 fps.
##
## i7 4500U:    Ubuntu16.04 -- 5 rtsp, CPU AI gets ~10.6 fps
##                          -- 5 rtsp, 1 NCS, no CPU AI gets ~10.7 fps
##                          -- 5 rtsp, CPU AI + 1 NCS gets ~20.1 fps
##                          -- 5 rtsp, CPU AI + 2 NCS gets ~24.4 fps, this is camera limited since each is set for 5 fps
##              Ubuntu18.04 -- 5 rtsp, CPU AI + 1 NCS ~19.5 fps.
##
## i5 4200U     Ubuntu16.04 -- 5 rtsp, CPU AI ~9.7 fps
##                          -- 5 rtsp, CPU AI + 1 NCS ~18.1 fps.
##
##
##
## 21MAR2019wbk
## i7 4 GHz quad core 6700K Desktop -- 2 Onvif, 5 rtsp, 1 mqtt  gets ~29.4 fps wtih one NCS and one CPU AI thread.
##
## 11MAR2019wbk
## i7 1 NCS, no CPU AI threads gives 10.6 fps with 4 rtsp streams for both roundrobin and one thread per stream sampling.
##
##
###
# AI_unified_mt.py
# 16JAN2019wbk derived from: AI_multi_threaded.unified.py
# Add option to capture images vis RTSP streams instead of jpeg snapshots.
# Move thread functions back into this module, the split just made things uglier, not better.
#
#
# AI_multi_threaded.unified.py
# 12NOV2018wbk derived from: onvif_AI_multi.py
# 5DEC2018wbk
# Unifiy to run on Linux or Windows, threading model on Windows is only minimally tested, 25FEB19 seems pretty solid on Win7 & Win10.
# move thread functions to a library file:  AIutils.py
# add automatic switch to openCV 3.2+ dnn module for CPU only MobilenetSSD
# Still works with python2.7 and python3.5 if Python 2.7 also has correct openCV version, last tested on Raspberry Pi
#
# 8DEC2018wbk
# Minor fixes and optimizations, better error handling -->  continue to run with camera failures (requires one thread per camera)
# On an i7 quad core with hyperhreading CPU AI -- 6 cameras -- 13 total threads: ~36 fps -- Camera limited, CPU_AI threads waiting for data
#    with a single Movidius NCS -- 6 cameras -- 8 total threads: ~10.7 fps -- AI limited, Onvif threads waiting for AI
# On i3 dual core Windows 7 -- 6 cameras -- 13 threads: ~6.7 fps -- AI limited.
#    same 6 cameras and this i3 system booted to Ubuntu-Mate 18.04: ~7.9 fps
#
# 11JAN2019wbk
# Make seperate queue for each camera unless roundrobin camera sampling specified.
# Allow mix of CPU and AI threads, default to 1 CPU thread in addition to any NCS threads
# On i3 with 3 cameras and 1 CPU thread got ~6.8 fps, adding 1 NCS stick got ~15.7 fps
#     using 2 CPU threads instead gets ~7.1 fps, adding -d 0 options only improves to ~7.8 fps
# On my i7 Desktop using only a single CPU thread gets ~20.5 fps
#
#
##
# onvif_AI_multi.py
# 10NOV2018wbk
# Many threads, one to read each camera, one for each NCS, and the main thread
# Initial version that seems to work pretty well.
# On i7 Desktop (Python 3.5) with 3 Onvif cameras and 1 NCS getting ~10.7 fps (5 threads total)
# with 3 cameras and 2 NCS getting ~20.5 fps (6 threads total)
# and with  3 cameras and 3 NCS getting ~29.5 fps (7 threads total)
# Trying a 4th NCS (10 threads) ~29 fps, clearly Camera limited.
#### Might still be useful for analyizing multiple sub-regions in a frame
#### instead of shrinking the entire frame (pretty much needed for better than 720p cameras).
# With 4 cameras round-robin, 1 NCS stick  (3 threads) ~7.6 fps
#### round-robin sampling is NOT recommended as it basically stops working with a network connection error or camera failure.
#
# 11NOV2018wbk
# remove restriction that all cameras need the same frame resolution.
#
# 13NOV2018wbk
# No NCS --> CPU AI:
# With 4 cameras round-robin, and 1 AI thread (3 threads) get ~5.8 fps on my heavily loaded i7 Desktop
# With 4 cameras round-robin, AI thread per camera (6 threads) ~ 9.9 fps -- apears Camera limited (Camera output queue never full, AI queues often empty)
# With 4 cameras, and thread per camera sampling (9 threads) ~30 fps -- apears Camera limited (Camera output queues never full, AI queues often empty)
#
# On lightly loaded i3 with round-robin camera sampling and one AI thread per camera (6 threads) get ~6.3 fps -- apears Camera limited
# On i3, (9 threads) gets only ~8 fps  -- appears AI limited (Cameara output queues often full, AI input queues never empty)
#
# 15AUG2019wbk
#   reduce wait time from 0.033 to 0.016 in results.put()
#   shorten results queue length from 2*Ncameras to 1 + Ncameras/2
#   reduce input queues to length 1
#   remove MQTTmode, one MQTT thread for all mqttCams is generally the way to go.
#   change detection default confidence to 0.70, verify confidence to 0.80
#
# 21NOV2019wbk
#   Remove round-robin sampling options.
# 
# 22AUG2019wbk
# Reorginize main loop to avoid imwrite() and imencode() if results are not going to be used.  Worthwhile on weak IOT hardware.
#
# 23AUG2019wbk
#   Some performance on i7-6700K Desktop: ./AI_OVmy.py -Nmqtt 15 -camMQTT i5a7 -d 2
#   -nNCS 1 ~25.2 fps   (NCS2)
#   -nNCS 2 ~23.4 fps   (NCS, thread0 ~11.7 fps, thread1 ~11.7 fps)
#   -nt 1   ~42.7 fps   (NCS thread ~9,8 fps, CPU thread ~32.9 fps)
#   -nNCS 0 -nt 1   ~42.7 fps
# Note that ~45 fps is processing every frame from all 15 Lorex DVR rtsp streams.
####
#
# 25SEPT2019wbk
#   i7-6700K Desktop, MobilenetSSD-v2_coco AI model
#   ./AI_OVmt.py -nNCS 0 -nt 1 -Nmqtt 8  with ./rtsp2mqttPdemand.py -rtsp 2UHD6HD.rtsp 2>/dev/null  running locally, 
#   2UHD (4K, 8 Mpixel), 6HD (1080p, 2K, 2 Mpixel) cameras 3 fps rtsp streams:  ~22.9 fps
#   ./AI_OVmt.py -nNCS 0 -nt 1 -rtsp 2UHD6HD.rtsp 2>/dev/null                   ~22.9 fps ==> no mqttCam inefficency on stong system.
#
# 15OCT2019wbk
# ./AI_OVmt.py -nNCS 0 -nt 1 -d 2 -cam spareURL.txt
# i7-6700K desktop:
#   OpenVINO OpenCV-4.1.0 SSD v2   :  ~20.2 fps
#   OpenCV 3.3.0 MobileNetSSD Caffe:  ~15.7 fps
# Note that on an i3-4025U MobilenetSSDv2 is ~5.5 fps with NCS, vs ~11.0 fps with CPU AI thread.

#
# 17OCT2019wbk -- Add syncronized wait to rtsp thread startup.
# NCS not very good with SSD v2:
# i7-6700K: ./AI_OVmt.py -d 2 -rtsp 2spareURL.rtsp --> ~6.1 fps
#           ./AI_OVmt.py -d 2 -nNCS 0 -nt 1 -rtsp 2spareURL.rtsp --> ~10.0 fps (basically every frame from these two 5 fps rtsp streams)


# determine OS platform
# TODO: if I ever get a Mac, enhance this to work there too.
# Note.  The Windows version has had only minimal testing on a friend's machine, CPU only AI, no NCS.
# Installing OpenVINO on Windows was more difficult than with Linux, hopefully newer
# versions fixed the issues (bummer if you have spaces in your user name!).
# I'm pretty sure all the places files are read need to have if __WIN__ changes to fix / vs \ path issues. 
# But on Linux, can now install OpenVINO with package managers like APT or YUM making it much easier.
# Worthwhile for the OpenCV version even if you don't have NCS/NCS2 sticks to get a good OpenCV version!
import platform
global __WIN__
if platform.system()[:3] != 'Lin':
    __WIN__ = True
else:
    __WIN__ = False


# Get python major version for
import sys
if sys.version_info.major < 3:
    print("Python version 3 is required.  Exiting ...")
    quit()
    

# import the necessary packages
if __WIN__ == False:
    import signal
from imutils.video import FPS
import argparse
import numpy as np
import cv2
import paho.mqtt.client as mqtt
import os
import datetime
import time
import requests
from PIL import Image
from io import BytesIO
# threading stuff
from queue import Queue
from threading import Lock, Thread


# mark start of this code in log file
print("")
print("********************************************************************")
currentDT = datetime.datetime.now()
print("**" + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
print("[INFO] using openCV-" + cv2.__version__)
#print("")

if cv2.__version__.find("openvino") < 0:     # we need "special" Intel OpenVINO version of OpenCV!
    print("Need Intel verison of OpenCV: 4.1.0-openvino or higher")
    print("No OpenVINO features will be used, thus no DNN_BACKEND_INFERENCE_ENGINE, NCS or NCS2 support!")
print("")
#    print(" installed is: " + cv2.__version__ + "exiting ...")
#    quit();


# *** System Globals
# these are write once in main() and read-only everywhere else, thus don't need syncronization
global QUIT
QUIT=False  # True exits main loop and all threads
global nextCamera
global CameraURL    # needs to still be global for non threaded call to grab an initial Onvif snapshot
global CamError     # needs to still be global for non threaded call to grab an initial Onvif snapshot
global Nrtsp
global Nonvif
global Ncameras
global nextCamera
global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
global UImode
global CameraToView
global sendAll
global saveAll
global confidence
global verifyConf
global noVerifyNeeded
global dbg
global subscribeTopic
subscribeTopic = "Alarm/#"  # topic controller publishes to to set AI operational modes
global Nmqtt
global mqttCamOffset
global inframe
global mqttFrameDrops
global mqttFrames


# these need syncronization
cameraLock = Lock()
global nextCamera
nextCamera = 0      # next camera queue for AI threads to use to grab a frame
rtspLock = Lock()   # make rtsp frame graps be atomic, seems openCV may not be completely thread safe.



# *** constants for MobileNet-SSD AI model
# frame dimensions should be sqaure for MobileNet-SSD
PREPROCESS_DIMS = (300, 300)
# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))  # from PyImageSearch example code
#COLORS[15] = (255,255,255)  # force person box to be white
#COLORS[15] = (0,0,255)  # force person box to be red
COLORS[15] = (0,255,0)  # force person box to be green



# *** Function definitions
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# Boilerplate code to setup signal handler for graceful shutdown on Linux
if __WIN__ is False:
    def sigint_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGINT, normal exit. -- ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        #quitQ.put(True)
        QUIT=True

    def sighup_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGHUP! ** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        #quitQ.put(True)
        QUIT=True

    def sigquit_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGQUIT! *** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        #quitQ.put(True)
        QUIT=True

    def sigterm_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGTERM! **** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        #quitQ.put(True)
        QUIT=True

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sighup_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)



#**********************************************************************************************************************
## MQTT callback functions
##
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    global subscribeTopic
    #print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.  -- straight from Paho-Mqtt docs!
    client.subscribe(subscribeTopic)


CameraToView=0
# The callback for when a PUBLISH message is received from the server, aka message from SUBSCRIBE topic.
AlarmMode="Audio"    # will be Email, Audio, or Idle  via MQTT from alarmboneServer
def on_message(client, userdata, msg):
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    global UImode
    global CameraToView
    global sendAll
    global saveAll
    if str(msg.topic) == "Alarm/MODE":          # Idle will not save detections, Audio & Notify are the same here
        currentDT = datetime.datetime.now()     # logfile entry
        AlarmMode = str(msg.payload.decode('utf-8'))
        print(str(msg.topic)+":  " + AlarmMode + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        return
    if str(msg.topic) == "Alarm/UImode":    # dashboard control Disable, Detections, Live exposes apparent node-red websocket bugs
        currentDT = datetime.datetime.now() # especially if browser is not on localhost, use sparingly, useful for camera setup.
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        UImode = int(msg.payload)
        return
    if str(msg.topic) == "Alarm/ViewCamera":    # dashboard control to select image to view
        currentDT = datetime.datetime.now()
        print(str(msg.topic)+": " + str(int(msg.payload)) + currentDT.strftime("   ... %Y-%m-%d %H:%M:%S"))
        CameraToView = int(msg.payload)
        return
    if str(msg.topic) == "Alarm/sendAll":   # sends all images not just detections as DetectionImageBuffer/CamN messages 
        currentDT = datetime.datetime.now()
        print(str(msg.topic)+":  " + str(msg.payload) + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        send = str(msg.payload)
        if send.count("True"):
            sendAll=True
        else:
            sendAll=False
        return
    if str(msg.topic) == "Alarm/saveAll":   # save all images, not just detections, will fill up drive fast!
        currentDT = datetime.datetime.now() # but helpful for troubleshooting
        print(str(msg.topic)+":  " + str(msg.payload) + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        save = str(msg.payload)
        if save.count("True"):
            saveAll=True
        else:
            saveAll=False
        return


def on_publish(client, userdata, mid):
    #print("mid: " + str(mid))      # don't think I need to care about this for now, print for initial tests
    pass


def on_disconnect(client, userdata, rc):
    if rc != 0:
        currentDT = datetime.datetime.now()
        print("Unexpected MQTT disconnection!" + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S  "), client)
    pass


# callbacks for mqttCam that can't be shared
def on_mqttCam_connect(client, camList, flags, rc):
     for camN in camList:
        client.subscribe("MQTTcam/"+str(camN), 0)


def on_mqttCam(client, camList, msg):
    global mqttCamOffset
    global inframe
    global mqttFrameDrops
    global mqttFrames
    global Nmqtt
    if msg.topic.startswith("MQTTcam/"):
        camNstr=msg.topic[len("MQTTcam/"):]    # get camera number as string
        if camNstr.isdecimal():
            camT = int(camNstr)
            if camT not in camList:
                currentDT = datetime.datetime.now()
                print("[Error! Invalid MQTTcam Camera number: " + str(camT) + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
                return
            for i in range(Nmqtt):
                if camT == camList[i]:
                    camN=i
                    break
        else:
            currentDT = datetime.datetime.now()
            print("[Error! Invalid MQTTcam message sub-topic: " + camNstr + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
            return
        # put input image into the camera's inframe queue
        try:
            mqttFrames[camN]+=1
            # thanks to @krambriw on the node-red user forum for clarifying this for me
            npimg=np.frombuffer(msg.payload, np.uint8)      # convert msg.payload to numpy array
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)   # decode image file into openCV image
            inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset), False) 
        except:
            mqttFrameDrops[camN]+=1
        try:
            client.publish(str("sendOne/" + str(camT)), "", 0, False)
        except Exception as e:
            print("pub error " + str(e))
        return


#**********************************************************************************************************************
## function to grab Onvif snapshot
# Some very slick, very high level code to grab a snapshot from an Onvif camera that supports snapshots.
# Finding the snapshot URL can be an issue.
# But I've some simple nodejs code to scan the local network for Onvif devices and print what it finds
def OnvifSnapshot(camera):
    global CamError
    global CameraURL
    try:
        r = requests.get(CameraURL[camera])
        i = Image.open(BytesIO(r.content))
        npimg = np.array(i)
        npimg=cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        CamError[camera]=False     # after geting a good frame, enable logging of next error
        return npimg
    except Exception as e:
        # this appears to fix the Besder camera problem where it drops out for minutes every 5-12 hours
        # likely issues here wtih round-robin sampling
        if not CamError[camera]:   # suppress the zillions of sequential error messages while it recovers
            currentDT = datetime.datetime.now()
            print('Onvif cam'+ str(camera) + ': ' + str(e) + CameraURL[camera] + ' --- ' + currentDT.strftime(" %Y-%m-%d_%H:%M:%S.%f"))
        frame = None
        CamError[camera]=True
        return None



# *** Thread functions
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# *** RTSP Sampling Thread
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def rtsp_thread(inframe, camn, URL):
    global QUIT
    global threadLock
    global threadsRunning
    ocnt=0
    Error=False
    Error2=False
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread" + str(camn) + " is starting..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling thread" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning += 1
    threadLock.release()
    while not QUIT:
         # grab the frame
        try:
            if Rcap.isOpened() and Rcap.grab():
                gotFrame, frame = Rcap.retrieve()
            else:
                frame = None
                if not Error:
                    Error=True
                    currentDT = datetime.datetime.now()
                    print('[Error!] RTSP Camera'+ str(camn) + ': ' + URL[0:33] + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                    print('*** Will close and re-open Camera' + str(camn) +' RTSP stream in attempt to recover.')
                # try closing the stream and reopeing it, I have one straight from China that does this error regularly
                Rcap.release()
                time.sleep(5.0)                   
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened():
                    if not Error2:
                        Error2=True                   
                        currentDT = datetime.datetime.now()
                        print('[Error2!] RTSP stream'+ str(camn) + ' re-open failed! $$$ ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                        print('*** Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(5.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera'+ str(camn) + ' has recovered: ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream'+ str(camn) + ': ' + str(e) + " " + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
        try:
            if frame is not None:
                if inframe.full():
                    [_,_]=inframe.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below     
                inframe.put((frame, camn), False)   # no block if queue full, go grab fresher frame
        except: # most likely queue is full
            ocnt+=1          
    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
    Rcap.release()
    print("RTSP stream sampling thread" + str(camn) + " is exiting, dropped frames " + str(ocnt) + " times.")



## *** ONVIF Sampling Thread ***
#******************************************************************************************************************
# Onvif camera sampling thread
def onvif_thread(inframe, camn, URL):
    global QUIT
    print("[INFO] ONVIF Camera" + str(camn) + " thread is running...")
    ocnt=0  # count of times inframe thread output queue was full (dropped frames)
    Error=False
    while not QUIT:
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
            if QUIT:
                break
            time.sleep(5.0)     # let other threads have more time while this camera recovers, which sometimes takes minutes
        try:
            if frame is not None and not QUIT:
                inframe.put((frame, camn), True, 0.200)
                ##time.sleep(sleepyTime)   # force thread switch, hopefully smoother sampling, 10Hz seems upper limit for snapshots
        except: # most likely queue is full
            if QUIT:
                break
            ocnt=ocnt+1
            ##time.sleep(sleepyTime)
            continue
    print("ONVIF Camera" + str(camn) + " thread is exiting, dropped frames " + str(ocnt) + " times.")



## *** OpenVINO NCS/NCS2 (aka MYRIAD) AI Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
def AI_thread(results, inframe, net, tnum, cameraLock, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dnnTarget):
    global nextCamera
    global QUIT
    global Ncameras
    global dbg
    global blobThreshold
    global SSDv1
    print("[INFO] OpenVINO " + dnnTarget + " AI thread" + str(tnum) + " is running...")
    fcnt=0
    waits=0
    drops=0
    # This is the crude 3.3.0 work-around to drop duplicate detections, seems real detections always vary a bit
    # I'm not sure later versions fix it, or if its confined to the Raspberry Pi camera module, but I've left it in here.
    #  removed and added back after seeing it again with SSDv2 and HummingbirtRight 1080p camera
    prevDetections=list()
    for i in range(Ncameras):
        prevDetections.append(0)
    if tnum > 0:
        dnnTarget = dnnTarget + str(tnum)
    cfps = FPS().start()
    while not QUIT:     
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
            ##if conf > confidence and idx == personIdx:
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
                        label = "{:.1f}%  ".format(conf * 100) + dnnTarget + "V"
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
                    results.put((img, cam, True, imgDT, dnnTarget +"V", boxPointsV), True, 1.0) # force zoom rejection file write
                results.put((image, cam, personDetected, imageDT, dnnTarget, boxPoints), True, 0.016)
        except:
            # presumably outptut queue was full, main thread too slow.
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("OpenVINO " + dnnTarget + " AI thread" + str(tnum) + ", waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))



# *** main()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main():
    global QUIT
    global UImode
    UImode=0    # controls if MQTT buffers of processed images from selected camera are sent as topic: ImageBuffer
    global sendAll
    global saveAll
    global confidence
    global verifyConf
    global noVerifyNeeded
    global dbg
    global subscribeTopic
    global CamError
    global CameraURL
    global Nonvif
    global Nrtsp
    global Nmqtt
    global mqttCamOffset
    global mqttFrames    
    global mqttFrameDrops
    global inframe
    global Ncameras
    global nextCamera
    global blobThreshold
    global SSDv1

    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()
    
    # must specify number of NCS sticks for OpenVINO, trying load in a try block and error, wrecks the system!
    ap.add_argument("-nNCS", "--nNCS", type=int, default=1, help="number of Myraid devices")
    # use Mobilenet-SSD Caffe model instead of Tensorflow Mobilenet-SSDv2_coco
    ap.add_argument("-SSDv1", "--SSDv1", action="store_true", help="Use original Mobilenet-SSD Caffe model for NCS")
    
    ap.add_argument("-c", "--confidence", type=float, default=.70, help="detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=.80, help="detection confidence for verification")
    ap.add_argument("-nvc", "--noVerifyConfidence", type=float, default=.98, help="initial detection confidence to skip verification")
    ap.add_argument("-dbg", "--debug", action="store_true", help="display images to debug detection verification thresholds")
    ap.add_argument("-blob", "--blobFilter", type=float, default=.20, help="reject detections that are more than this fraction of the frame")
    
    # number of software (CPU only) AI threads, always have one thread per installed NCS stick
    # os.cpu_count() will return the cores/hyperthreads, but appaentyl breaks Python 2.7 compatability
    ap.add_argument("-nt", "--nAIcpuThreads", type=int, default=0, help="0 --> no CPU AI thread, >0 --> N threads")

    # specify MQTT broker for camera images via MQTT, if not "localhost"
    ap.add_argument("-camMQTT", "--mqttCameraBroker", default="localhost", help="name or IP of MQTTcam/# message broker")
    # number of MQTT cameras published as Topic: MQTTcam/N, subscribed here as Topic: MQTTcam/#, Cams numbered 0 to N-1
    ap.add_argument("-Nmqtt", "--NmqttCams", type=int, default=0, help="number of MQTT cameras published as Topic: MQTTcam/N,  Cams numbered 0 to N-1")
    # alternate, specify a list of camera numbers
    ap.add_argument("-camList", "--mqttCamList", type=int, nargs='+', help="list of MQTTcam/N subscription topic numbers,  cam/N numbered from 0 to Nmqtt-1.")
    
    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="cameraURL.rtsp", help="path to file containing rtsp camera stream URLs")
        
    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="cameraURL.txt", help="path to file containing http camera jpeg image URLs")
    
    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", type=int, default=2,
        help="display images on host screen, 0=no display, 1=detections only, 2=live & detections")

    # specify MQTT broker
    ap.add_argument("-mqtt", "--mqttBroker", default="localhost", help="name or IP of MQTT Broker")
    
    # specify display width and height
    ap.add_argument("-dw", "--displayWidth", type=int, default=1920, help="host display Width in pixels, default=1920")
    ap.add_argument("-dh", "--displayHeight", type=int, default=1080, help="host display Height in pixels, default=1080")

    # specify host display width and height of camera image
    ap.add_argument("-iw", "--imwinWidth", type=int, default=640, help="camera image host display window Width in pixels, default=640")
    ap.add_argument("-ih", "--imwinHeight", type=int, default=360, help="camera image host display window Height in pixels, default=360")
    
    # enable local save of detections on AI host, useful if node-red notification code is not being used   
    ap.add_argument("-ls", "--localSave", action="store_true", help="save detection images on local AI host")
    # specify file path of location to same detection images on the localhost
    ap.add_argument("-sp", "--savePath", default="", help="path to location for saving detection images, default ~/detect")
    # save all processed images, fills disk quickly, really slows things down, but useful for test/debug
    ap.add_argument("-save", "--saveAll", action="store_true", help="save all images not just detections on host filesystem, for test/debug")
   
    args = vars(ap.parse_args())


    # set variables from command line auguments or defaults
    confidence = args["confidence"]
    verifyConf = args["verifyConfidence"]
    noVerifyNeeded = args["noVerifyConfidence"]
    blobThreshold = args["blobFilter"]
    dbg=args["debug"]
    nCPUthreads = args["nAIcpuThreads"]
    Nmqtt = args["NmqttCams"]
    camList=args["mqttCamList"]
    if camList is not None:
        Nmqtt=len(camList)
    elif Nmqtt>0:
        camList=[]
        for i in range(Nmqtt):
            camList.append(i)
    dispMode = args["display"]
    if dispMode > 2:
        displayMode=2
    CAMERAS = args["cameraURLs"]
    RTSP = args["rtspURLs"]
    nNCS = args["nNCS"]
    SSDv1 = args["SSDv1"]
    MQTTserver = args["mqttBroker"]
    MQTTcameraServer = args["mqttCameraBroker"]
    displayWidth = args["displayWidth"]
    displayHeight = args["displayHeight"]
    imwinWidth = args["imwinWidth"]
    imwinHeight = args["imwinHeight"]
    savePath = args["savePath"]
    saveAll = args["saveAll"]
    localSave = args["localSave"]
    if saveAll:
        localSave = True
    

    # *** get Onvif camera URLs
    # cameraURL.txt file can be created by first running the nodejs program (requires node-onvif be installed):
    # nodejs onvif_discover.js
    #
    # This code does not really use any Onvif features, Onvif compatability is useful to "automate" getting  URLs used to grab snapshots.
    # Any camera that returns a jpeg image from a web request to a static URL should work.
    try:
        CameraURL=[line.rstrip() for line in open(CAMERAS)]    # force file not found
        Nonvif=len(CameraURL)
        CamError=list()
        for i in range(Nonvif):    # set up error flags for each camera
            CamError.append(False)
        print("[INFO] " + str(Nonvif) + " http Onvif snapshot threads will be created.")
    except:
        # fallback to trying cameras in my test setup
        print("[INFO] No " + str(CAMERAS) + " file.  No Onvif snapshot threads will be created.")
        Nonvif=0
    Ncameras=Nonvif


    # *** get rtsp URLs
    try:
        rtspURL=[line.rstrip() for line in open(RTSP)]
        Nrtsp=len(rtspURL)
        rtspError=list()
        for i in range(Nrtsp):    # set up error flags for each camera
            rtspError.append(False)
        print("[INFO] " + str(Nrtsp) + " rtsp stream threads will be created.")
    except:
        # fallback to trying cameras in my test setup
        print("[INFO] No " + str(RTSP) + " file.  No rtsp stream threads will be created.")
        Nrtsp=0
    Ncameras+=Nrtsp


    # *** setup path to save AI detection images
    if savePath == "":
        detectPath= os.getcwd()
        detectPath=detectPath + "/detect"
        if os.path.exists(detectPath) == False and localSave:
            os.mkdir(detectPath)
    else:
        detectPath=savePath
        if os.path.exists(detectPath) == False:
            print(" Path to location to save detection images must exist!  Exiting ...")
            quit()


    # *** allocate queues
    # we simply make one queue for each camera, rtsp stream, and MQTTcamera
    QDEPTH = 2      # small values improve latency
##    QDEPTH = 1      # small values improve latency
    print("[INFO] allocating camera and stream image queues...")
    mqttCamOffset = Ncameras
    mqttFrameDrops = 0
    mqttFrames = 0
    Ncameras+=Nmqtt     # I generally expect Nmqtt to be zero if Ncameras is not zero at this point, but its not necessary
    if Ncameras == 0:
        print("[INFO] No Cameras, rtsp Streams, or MQTT image inputs specified!  Exiting...")
        quit()
    if Nmqtt > 0:
        print("[INFO] allocating " +str(Nmqtt) + " MQTT image queues...")
    inframe = list()
    results = Queue(int(Ncameras/2)+1)
    for i in range(Ncameras):
            inframe.append(Queue(QDEPTH))


    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])


    # *** setup display windows if necessary
    # mostly for initial setup and testing, not worth a lot of effort at the moment
    if dispMode > 2:
        displayMode=2
    if dispMode > 0:
        if Nonvif > 0:
            print("[INFO] setting up Onvif camera image windows ...")
            for i in range(Nonvif):
                if dispMode == 1:
                    name=str("Detect_" + str(i))
                else:
                    name=str("Live_" + str(i))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                frame=OnvifSnapshot(i)
                if frame is None:
                    print("*** Bad Camera! *** URL: " + str(CameraURL[i]))
                    CamError[i]=True
                    continue
                cv2.imshow(name, cv2.resize(frame, (imwinWidth, imwinHeight)))
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                if dispMode == 1:
                    name=str("Detect_" + str(i+Nonvif))
                else:
                    name=str("Live_" + str(i+Nonvif))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nmqtt > 0:
            print("[INFO] setting up MQTT camera image windows ...")
            for i in range(Nmqtt):
                if dispMode == 1:
                    name=str("Detect_" + str(i+mqttCamOffset))
                else:
                    name=str("Live_" + str(i+mqttCamOffset))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)
               

        # *** move windows into tiled grid
        top=2
        left=2
        ##left=1900 # my 4K display
        ##displayHeight=1900
        Xshift=imwinWidth+3
        Yshift=imwinHeight+28
        Nrows=int(displayHeight/imwinHeight)    
        for i in range(Ncameras):
            if dispMode == 1:
                name=str("Detect_" + str(i))
            else:
                name=str("Live_" + str(i))
            col=int(i/Nrows)
            row=i%Nrows
            cv2.moveWindow(name, left+col*Xshift, top+row*Yshift)
                    

    # *** connect to MQTT broker
    print("[INFO] connecting to MQTT " + MQTTserver + " broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.will_set("AI/Status", "Python AI has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()

    # *** MQTT send a blank image to the dashboard UI
    print("[INFO] Clearing dashboard ...")
    client.publish("ImageBuffer/!AI has Started.", bytearray(img_as_jpg), 0, False)

    # *** Open second MQTT client thread for MQTTcam/# messages
    # Requires rtsp2mqttPdemand.py mqtt camera source
    if Nmqtt > 0:
        mqttFrameDrops=[]
        mqttFrames=[]
        mqttCam=list()
        print("[INFO] connecting to " + MQTTcameraServer + " broker for MQTT cameras...")
        print("INFO all MQTT cameras will be handled in a single thread.")
        for i in camList:
            mqttFrameDrops.append(0)
            mqttFrames.append(0)
        mqttCam = mqtt.Client(userdata=camList, clean_session=True)
        mqttCam.on_connect = on_mqttCam_connect
        mqttCam.on_message = on_mqttCam
        mqttCam.on_publish = on_publish
        mqttCam.on_disconnect = on_disconnect
        mqttCam.connect(MQTTcameraServer, 1883, 60)
        mqttCam.loop_start()
        time.sleep(0.1)     # force thread dispatch
        for i in camList:
            mqttCam.publish(str("sendOne/" + str(i)), "", 0, False)   # start messages

    

    # *** start camera reading threads
    o = list()
    if Nonvif > 0:
        print("[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        for i in range(Nonvif):
            o.append(Thread(target=onvif_thread, args=(inframe[i], i, CameraURL[i])))
            o[i].start()
    if Nrtsp > 0:
        global threadLock
        global threadsRunning
        threadLock = Lock()
        threadsRunning = 0
        print("[INFO] starting " + str(Nrtsp) + " RTSP Camera Sampling Threads ...")
        for i in range(Nrtsp):
            o.append(Thread(target=rtsp_thread, args=(inframe[i+Nonvif], i+Nonvif, rtspURL[i])))
            o[i+Nonvif].start()
        while threadsRunning < Nrtsp:
            time.sleep(0.5)
        print("[INFO] All " + str(Nrtsp) + " RTSP Camera Sampling Threads are running.")


    # *** setup and start Myriad OpenVINO
    if nNCS > 0:
      if cv2.__version__.find("openvino") > 0:
        if SSDv1:
            print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO Myriad NCS/NCS2 AI threads...")
            OVstr = "OVncs"
        else:
            print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 model for OpenVINO Myriad NCS/NCS2 AI threads...")
            OVstr = "SSDv2ncs"
        netOV=list()
        for i in range(nNCS):
            print("... loading model...")
            if SSDv1:
                netOV.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
            else:
                netOV.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2coco.xml", "mobilenet_ssd_v2/MobilenetSSDv2coco.bin"))
            netOV[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            netOV[i].setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)  # specify the target device as the Myriad processor on the NCS
        # *** start OpenVINO AI threads
        OVt = list()
        print("[INFO] starting " + str(nNCS) + " OpenVINO Myriad NCS/NCS2 AI Threads ...")
        for i in range(nNCS):
            OVt.append(Thread(target=AI_thread, 
                args=(results, inframe, netOV[i], i, cameraLock, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, OVstr)))
            OVt[i].start()
      else:
        print("[ERROR!] OpenVINO version of openCV is not active, check $PYTHONPATH")
        print(" No MYRIAD (NCS/NCS2) OpenVINO threads will be created!")
        nNCS = 0
        if nCPUthreads == 0:
            print("[INFO] No NCS threads specified, forcing one CPU AI thread.")
            nCPUthreads=1   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+


    # ** setup and start CPU AI threads, usually only one makes sense.
    if nCPUthreads > 0:
        net=list()
        if cv2.__version__.find("openvino") > 0:
            if SSDv1:
                print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO CPU AI threads...")
                dnnTarget="OVcpu"
                for i in range(nCPUthreads):
                    net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
                    net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                    net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)          
            else:
                print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP32 model for OpenVINO CPU AI threads...")
                dnnTarget = "SSDv2cpu"
                for i in range(nCPUthreads):
                    # newer OpenVINO versions can use the FP16 model same as Myraid target, this saves on distribution package size
                    # but complicates my life more by requiring all test systems to be updated.  I tested and it works but I'll stick
                    # with FP32 model for CPU AI for the time being.
                    net.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.bin"))
                    net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                    net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)          
        else:
            print("[INFO] OpenVINO not active.")
            print("[INFO] loading Caffe Mobilenet-SSD model for ocvdnn CPU AI threads...")
            dnnTarget="ocvCPU"
            for i in range(nCPUthreads):
                net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))       
        # *** start CPU AI threads
        CPUt = list()
        if cv2.__version__.find("openvino") > 0:
            print("[INFO] starting " + str(nCPUthreads) + " OpenVINO CPU AI Threads ...")
        else:
            print("[INFO] starting " + str(nCPUthreads) + " openCV dnn module CPU AI Threads ...")
        for i in range(nCPUthreads):
            CPUt.append(Thread(target=AI_thread, 
                 args=(results, inframe, net[i], i, cameraLock, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dnnTarget)))
            CPUt[i].start()
    
 
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    waitCnt=0
    currentDT = datetime.datetime.now()
    prevUImode=UImode
    print("[INFO] AI/Status: Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.publish("AI/Status", "Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    #start the FPS counter
    print("[INFO] starting the FPS counter ...")
    fps = FPS().start()
    while not QUIT:
        try:
            try:
                (img, cami, personDetected, dt, ai, bp) = results.get(True,0.100)
            except:
                waitCnt+=1
                img=None
                aliveCount = (aliveCount+1) % 200   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                continue
            if img is not None:
                fps.update()    # update the FPS counter
                # setup for file saving
                currentDT = datetime.datetime.now()
                folder=dt.strftime("%Y-%m-%d")
                filename=dt.strftime("%H_%M_%S.%f")
                filename=filename[:-5] + "_" + ai  #just keep tenths, append AI source
                if localSave:
                    if __WIN__ is False:
                        lfolder=str(detectPath + "/" + folder)
                    else:
                        lfolder=str(detectPath + "\\" + folder)
                    if os.path.exists(lfolder) == False:
                        os.mkdir(lfolder)
                    if __WIN__ is False:
                        if personDetected:
                            outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                        else:
                            outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                    else:
                        if personDetected:
                            outName=str(lfolder + "\\" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                        else:
                            outName=str(detectPath + "\\" + filename + "_" + "Cam" + str(cami) +".jpg")
                    if (personDetected and not AlarmMode.count("Idle")) or saveAll:  # save detected image
                        cv2.imwrite(outName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])     
                if personDetected:
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    # for sending image as mqtt buffer
                    if retv:
                        oName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                        oName=oName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                        client.publish(str(oName), bytearray(img_as_jpg), 0, False)
##                        print(oName)  # log detections
                    else:
                        print("[INFO] conversion of np array to jpg in buffer failed!")
                        continue
                # save image for live display in dashboard
                if ((CameraToView == cami) and (UImode == 1 or (UImode == 2 and personDetected))) or (UImode ==3 and personDetected):
                    if personDetected:
                        topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                    else:
                        retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        if retv:
                            topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +".jpg")
                        else:
                            print("[INFO] conversion of np array to jpg in buffer failed!")
                            continue
                    client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                # display the frame to the screen if enabled, in normal usage display is 0 (off)
                if dispMode > 0:
                    if personDetected and dispMode == 1:
                        name=str("Detect_" + str(cami))
                        cv2.imshow(name, cv2.resize(img, (imwinWidth, imwinHeight)))
                    elif dispMode == 2:
                        name=str("Live_" + str(cami))
                        cv2.imshow(name, cv2.resize(img, (imwinWidth, imwinHeight)))
            if dispMode > 0:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): # if the `q` key was pressed, break from the loop
                    QUIT=True   # exit main loop
                    continue
            if prevUImode != UImode:
                img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
                img[:,:] = (154,127,100)
                retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                client.publish("ImageBuffer/!AI Mode Changed.", bytearray(img_as_jpg), 0, False)
                prevUImode=UImode
            aliveCount = (aliveCount+1) % 200
            if aliveCount == 0:
                client.publish("AmAlive", "true", 0, False)
        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            QUIT=True   # exit main loop
            continue
        except Exception as e:
            currentDT = datetime.datetime.now()
            print(" **** Main Loop Error: " + str(e)  + currentDT.strftime(" -- %Y-%m-%d %H:%M:%S.%f"))
            excount=excount+1
            if excount <= 3:
                continue    # hope for the best!
            else:
                break       # give up! Hope watchdog gets us going again!
    #end of while not QUIT  loop


    # *** Clean up for program exit
    fps.stop()    # stop the FPS counter timer
    currentDT = datetime.datetime.now()
    print("Program Exit signal received:" + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    # display FPS information
    print("[INFO] Run elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] AI processing approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] Frames processed by AI system: " + str(fps._numFrames))
    print("[INFO] Main looped waited for results: " + str(waitCnt) + " times.")
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    print("AI/Status: Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))


    # stop cameras 
    if Nmqtt > 0:
        mqttCam.disconnect()
        mqttCam.loop_stop()
        for i in range(Nmqtt):
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))

    if Nonvif > 0:
        for i in range(Nonvif):
            o[i].join()
    if Nrtsp > 0:
        for i in range(Nrtsp):
            o[i+Nonvif].join()
    print("[INFO] All Camera Threads have exited ...")

    
    # wait for threads to exit
    if nCPUthreads > 0:
        for i in range(nCPUthreads):
            if not results.empty():
                (_, _, _, _, _, _) = results.get(False)
            CPUt[i].join()
        print("[INFO] All CPU AI Threads have exited ...")

    if nNCS > 0:
        for i in range(nNCS):
            if not results.empty():
                (_, _, _, _, _, _) = results.get(False)
            OVt[i].join()
        print("[INFO] All OpenVINO NCS AI Threads have exited ...")

    
    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()


    # Send a blank image the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros((imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (32,32,32)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    client.publish("ImageBuffer/!AI has Exited.", bytearray(img_as_jpg), 0, False)
    time.sleep(1.0)


    # clean up MQTT
    client.disconnect()     # normal exit, Will message should not be sent.
    currentDT = datetime.datetime.now()
    print("Stopping MQTT Threads." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    time.sleep(1.0)
    client.loop_stop()      ### Stop MQTT thread

    # bye-bye
    currentDT = datetime.datetime.now()
    print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    

# python boilerplate
if __name__ == '__main__':
    main()


