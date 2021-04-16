#!/usr/bin/env python3
#
### AI_dev.py 13JUL2019wbk
#
### 16APR2021wbk
#       Modified TPU support to try the "legacy" edgetpu API and if its not found try the new PyCoral API
#       Tested on Ubuntu 20.04 i3-4025 CPU with PyCoral and the new MPCIe TPU module (< half the cost of USB3 TPU)
#       Verified on Ubuntu 16.04 i7 desktop using "legacy" edgeTPU API
### derived from AI_OVmt.py & AI_Coral.py
## 21JUN2019wbk Some initial test results for AI_Coral.py, 1 Coral USB stick on USB2 i7 4 GHz quad core 6700K Desktop, 15 mqtt camera inputs,
### with rtsp2mqtt running on the same system (30 windows displayed on my 4K monitor 15 inputs, 15 AI results)
### with verification debugging enabled: ~33.3 fps.   This is a bit better than 2 fps per camera!
###
### This mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite model seems significantly more sensitive detecting "person"
### in HD (1920x1080) images than are the MobileNetSSD caffe models used with the NCS/NCS2 and CPU OpenCV dnn module.
#
### 5JUL2019wbk  Add back OpenVINO NCS support and CPU OpenCV dnn module AI so I only have one program to move forward with.
### Move Coral, OpenVINO, and OpenCV dnn module AI threads to seperate modules.
###
### Note about these tests, since I added the "zoom and verify" the fps is lowered when there are many detecitons, I ran
### these tests when there was essentially no activity -- noting to zoom in to verify.
###
### Tests with 15 mqtt cameras (i7-4500U rtsp2mqtt converter system pushes ~36 fps) on my i7-6700K desktop:
### Getting approx linear speed up with multiple NCS sticks ~11.7 fps with one, ~21.2 with two (one was on USB2 port).
### I only have a single NCS2 stick, but it gets ~26.0 fps, mixing NCS & NCS2 got ~34.6 fps
### The Coral TPU gets ~31.5 fps with openCV-4.1.0-openvino, ~36.9 with openCV-3.3.0 compiled locally on this host.
### 20JUL2019wbk improvements to rtsp2mqttP.py has it pushing ~59.2 fps to my i7-6700K with Coral TPU!  ~75 fps is max from the Lorex DVR rtsp.
### Remove the --saveAll, --sendAll and detection only display code, don't really find them usefull.
### Add arguement to specify list of mqtt cameras to subscribe to, instead of only sequential from 0 to Nmqtt-1.
#
### 11JUL2019wbk moved onvif and rtsp threads to seperate modules.
#
### 12JUL2019wbk  Add back support for NCS V1 SDK, the OpenVINO version of openCV has poor rtsp decode performance, so
### it might sometimes be better to use an NCS or two wtih SDK V1 instead of an NCS2 with OpenVINO.
#
# TODO:
#   1) Try to get MobileNet_SSD_v2 for OpenVINO CPU/NCS/NCS2 and OpenCV dnn module.
##     Done, for NCS/NCS2 need to "optimize" an FP32 version for the CPU.
##     Frame rate for TensorFlow SSD_v2 is about half that of Caffe SSD_v1 on NCS/NCS2  false detections seem fewer reruning bogus detection images.
##     CPU TensorFlow SSD_v2 vs. Caffe SSD_v1 is minimal, ~23.5 fps vs. ~26.4 on i7-6700k desktop/
#   2) Support multiple Coral sticks?  Probably not,it may be cam input and/or results output limited as it is, but its worth adding
#      "probe" for installed TPU and option flag for noTPU as done with NCS SDKv1 code.
#
# 15AUG2019wbk
#   reduce wait time from 0.033 to 0.016 in results.put(), done in the thread functions
#   shorten results queue length from 2*Ncameras to 1 + Ncameras/2
#   reduce input queues to length 1
#   change detection default confidence to 0.70, verify confidence to 0.80
#
# 20AUG2019wbk
#   Return box points for detection on results queue, requires changes to all AI thread functions.
#   Abuse MQTT broker by sending filename and box points as message topic for the jpeg image buffer.
#   Make local save of detections be a command line option, default to False, but for IOT systems having remote MQTT broker system
# running node-red save the detections and do notifications is the prefered way, although its also possible to run the broker and
# node-red on the AI host.
##
# 23AUG2019wbk
#   Some performance on i7-6700K Desktop: ./AI_dev.py -Nmqtt 15 -camMQTT i5ai -d 2
#   -nTPU 1 ~42.7 fps
#   -nNCS 1 ~25.0 fps   (NCS2)
#   -nNCS 2 ~23.0 fps   (NCS, thread0 ~11.6 fps, thread1 ~11.4 fps)
#   -nt 1   ~40.4 fps
# Note that ~45 fps is processing every frame from all 15 Lorex DVR 3 fps rtsp streams.
#
# 17OCT2019wbk  rtsp2mqttTdemand.py all on i7-6700K desktop
# ./AI_dev.py -nNCS 0 -nt 1 -d 2 -Nmqtt 15 --> ~36.5 fps
# ./AI_dev.py -nNCS 0 -nt 0 -nTPU 1 -d 2 -Nmqtt 15 --> ~52.9 fps (obviously processing some duplicate frames)
##
##
# 16SEP2019wbk
# Jetson Nano, -nTPU 1 -d 0 -mqtt kahuna.local (i7-6700K) -rtsp xxx.rtsp 2>/dev/null (surpress rtsp decode warnings)
#   1 1080p and 1 4K RTSP 3 fps streams (decoded on Jetson):  ~5.8 fps (basically every frame processed by AI)
#   3 1080p and 1 4K                                       :  ~11.4 fps (pretty close to procesing every frame)
#   4 1080p                                                :  ~11.8 fps
#   5 1080p and 1 4K                                       :  ~17.3 fps (again, almost processing every frame!)
#   7 1080p and 1 4K                                       :  ~22.7 fps ( 24 fps would be every frame)
#   8 1080p                                                :  ~23.7 fps
#   2 4k and 2 1080p                                       :  ~11.8 fps (12 would be every frame)
#   5 4k                                                   :  ~14.7 fps (15 would be every frame)
#   5 4K and 3 1080p                                       :  ~21.6 fps
#   6 4K and 2 1080p                                       :  ~18.7 fps
#   6 4K                                                   :  ~16.8 fps (6 4K streams seems to be a tad too much), repeat ~16.9 fps.
#
# Jetson Nano, -camList 0 1 2 3 -camMQTT i5ai.local -nTPU 1 -d 0 -mqtt kahuna.local 2>/dev/null,
#   cams 0 & 1 are 4K: ~8.8 fps, some inefficiencey in MQTTcam code with remote decoding.  Network issues?
#
#   ./rtsp2mqttPdemand.py  -rtsp 5UHD.rtsp 2>/dev/null   (on Jetson Nano)
#   ./AI_dev.py -nTPU 1 -d 0 -Nmqtt 5 -mqtt kahuna.local -camMQTT localhost
#   5 4K cameras    : ~3.7 fps ==> very poor!
#   ./rtsp2mqttPdemand.py  -rtsp 8HD.rtsp 2>/dev/null   (on Jetson Nano)
#   ./AI_dev.py -nTPU 1 -d 0 -Nmqtt 8 -mqtt kahuna.local -camMQTT localhost
#   8 1080p cameras : ~18.5 fps, ==> not so bad, but still inferior to AI_dev.py -rtsp 8HD.rtsp
##
# NOTE: the above performace tests are with 1080p HD camera streams.  The rtsp2mqtt.py "server" and "mqtt cams" looked like a
# good solution.  Unfortunately it didn't scale well at all when I upgraded to 4K UHD cameras
##
#
# Pi4B 2GB RAM, -nTPU 1 -d 0 -mqtt kahuna -rtsp xxx.rtsp 2>/dev/null
#   1 1080p and 1 4K RTSP 3 fps streams (decoded on Pi4B):  ~5.8 fps,   processing about every frame
#   2 1080p and 2 4K                                     :  ~8.5 fps,   two 4K might be too much for the Pi4, would 4GB RAM like the Nano help?
#   3 1080p and 1 4K                                     :  ~11.8 fps,  1 4K and 3 1080p, about every frame
#   5 1080p and 1 4K                                     :  ~11.6 fps
#   6 1080p                                              :  ~17.4 fps,  is 4K is too much for the pi4?
#   8 1080p                                              :  ~13.9 fps,  seems 6 1080p rtsp streams is about the optimum.
#   2 4K                                                 :  ~6.0 fps
#
# Pi4B 4GB Ram, -nTPU 1 -d 0 -mqtt kahuna -rtsp xxx.rtsp 2>/dev/null
#   6 4K        : ~2.0 fps ==> far inferior to Jetson Namo for UHD streams.
#   8 1080p     : ~14.0 fps ==> definitley not RAM issue, got ~13.9 with 2GB.
#   3 4K        : ~4.9 fps    Pi4 is not good with 4K
#
# Pi4B 2GB RAM,  -camList 0 1 2 3 -camMQTT i5ai -nTPU 1 -d 0 -mqtt kahuna 2>/dev/null, cams 0 & 1 are 4K: ~6.1 fps
# Pi4B 2GB RAM,  -camList 2 3 4 5 -camMQTT i5ai -nTPU 1 -d 0 -mqtt kahuna 2>/dev/null,                  : ~11.0 fps
# Pi4B 2GB RAM,  -camList 2 3 4 5 6 7 -camMQTT i5ai -nTPU 1 -d 0 -mqtt kahuna 2>/dev/null,              : ~12.7 fps some inefficiencey in my MQTTcam code
##
# 14OCT2019wbk
# At this point, using rtsp2mqtt.py is not recommended, as generally get better performance on IOT class machines with native
# RTSP stream decoding.  This is exactly the situation I was trying to improve, for now, rate as fail.
#
#
# 17OCT2019wbk -- Add syncronized wait to rtsp thread startup.
#
# 5DEC2019wbk some TPU Pi4B tests with rtsp cameras, 3fps per stream:
# 4 UHD (4K)  :     ~2.8 fps
# 4 HD (1080p):     ~11.8 fps (basically processing every frame)
# 2 UHD 2 HD  :     ~6.7 fps (Pi4B struggles with 4K streams)
# 5 HD        :     ~14.7 fps (basically processing every frame)
# 6 HD        :     ~15.0 fps, -d 0 (no display) ~16.7 fps
# 8 HD        :    ~11.6 fps, -d 0 ~14.6 fps
#
## 6DEC2019wbk Some UHD & HD rtsp tests on Jetson Nano with TPU
# 5 UHD (4K)         :  ~14.6 fps (effectively processing every frame!)
# 5 UHD 3 HD         :  ~10.3 fps, jumps to ~19.1 fps if -d 0 option used (no live image display)
# 4 UHD 4 HD         :  ~16.3 fps, ~22.5 fps with -d 0 option
# 5 UHD 10 HD (1080p):  ~4.4 fps, ~7.6 fps with -d 0 option (totally overloaded, get ~39 fps with running on i7-4500U MiniPC)
#
## 7DEC2019wbk Coral Development Board (built-in TPU)
# 4 HD (1080p)        : ~11.9 fps (basically processing every frame)
# 2 UHD 2 HD          : ~11.7 fps
# 2 UHD 3 HD          : ~14.6 fps
# 2 UHD 4 HD          : ~12.3 fps, -d 0 (no display) ~16.7 fps
# 3 UHD               : ~8.8 fps (basically processing every frame)
# 4 UHD               : ~0.1 fps on short run, System locks up eventually!
# 3 UHD 2 HD          : ~0.27 fps Hopelessly overloaded, extremely slugglish.
# 6 HD                : ~17.9 fps
# 8 HD                : ~16.8 fps, -d 0 (no display) ~20.5 fps
#
## 10DEC2019wbk
# Increase queue depth to 2, test if queue full, read and discard oldest to make room for newest
#
## 11DEC2019wbk, add PiCamera Module support, change some command argument defaults and names.
## 27DEC2019wbk, tested PiCamera Module support on Pi3B with NCS and OpenVINO:
#  ./AI_dev.py -nNCS 1 -pi -ls --> get ~3.8 fps
# And Coral TPU:
#  ./AI_dev.py -nTPU 1 -pi -ls --> get ~8.0 fps



# import the necessary packages
import sys
import signal
from imutils.video import FPS
import argparse
import numpy as np
import cv2
import paho.mqtt.client as mqtt
import os
import time
import datetime
import requests
from PIL import Image
from io import BytesIO

# threading stuff
from queue import Queue
from threading import Lock, Thread


# *** System Globals
# these are write once in main() and read-only everywhere else, thus don't need syncronization
global QUIT
QUIT=False  # True exits main loop and all threads
global Nrtsp
global Nonvif
global Ncameras
global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
global UImode
global CameraToView
global subscribeTopic
subscribeTopic = "Alarm/#"  # topic controller publishes to to set AI operational modes
global Nmqtt
global mqttCamOffset
global inframe
global mqttFrameDrops
global mqttFrames
global mqttCamsOneThread
# this variable to distribute queued data to the AI threads needs syncronization
global nextCamera
nextCamera = 0      # next camera queue for AI threads to use to grab a frame
cameraLock = Lock()


# *** constants for MobileNet-SSD & MobileNet-SSD_V2  AI models
# frame dimensions should be sqaure for MobileNet-SSD
PREPROCESS_DIMS = (300, 300)


if 1:
    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()

    ap.add_argument("-c", "--confidence", type=float, default=.70, help="detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=.80, help="detection confidence for verification")
    ap.add_argument("-nvc", "--noVerifyConfidence", type=float, default=.98, help="initial detection confidence to skip verification")
    ap.add_argument("-dbg", "--debug", action="store_true", help="display images to debug detection verification thresholds")
    ap.add_argument("-blob", "--blobFilter", type=float, default=.20, help="reject detections that are more than this fraction of the frame")

    # specify number of Coral TPU sticks
    ap.add_argument("-nTPU", "--nTPU", type=int, default=0, help="number of Coral TPU devices")

    # must specify number of NCS sticks for OpenVINO, trying load in a try block and error, wrecks the system!
    ap.add_argument("-nNCS", "--nNCS", type=int, default=0, help="number of Myraid devices")
    # Use NCS SDK V1, if OpenVINO is specified this setting will be ignored and OpenVINO used.
    ap.add_argument("-sdk", "--sdkV1", action="store_true", help="use NCS sdkV1 instead of OpenVINO")
    # use Mobilenet-SSD Caffe model instead of Tensorflow Mobilenet-SSDv2_coco
    ap.add_argument("-SSDv1", "--SSDv1", action="store_true", help="Use original Mobilenet-SSD Caffe model for NCS & OVcpu")

    # use one mqtt thread for all cameras instead of one mqtt thread per mqtt camera
    ap.add_argument("-mqttMode", "--mqttCamOneThread", action="store_true", help="Use one mqtt thread for all mqtt cameras")
    ap.add_argument("-mqttDemand", "--mqttDemand", action="store_true", help="Use sendOne/N handshake for MQTT cameras")

    # number of software (CPU only) AI threads, always have one thread per installed NCS stick
    ap.add_argument("-nt", "--nAIcpuThreads", type=int, default=0, help="0 --> no CPU AI thread, >0 --> N threads")
    ap.add_argument("-GPU", "--GPU", action="store_true", help="use GPU insteas of CPU AI thread")

    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="cameraURL.rtsp", help="path to file containing rtsp camera stream URLs")

    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="cameraURL.txt", help="path to file containing http camera jpeg image URLs")

    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", type=int, default=1,
        help="display images on host screen, 0=no display, 1=live display")

    # specify MQTT broker
    ap.add_argument("-mqtt", "--mqttBroker", default="localhost", help="name or IP of MQTT Broker")

    # specify MQTT broker for camera images via MQTT, if not "localhost"
    ap.add_argument("-camMQTT", "--mqttCameraBroker", default="localhost", help="name or IP of MQTTcam/# message broker")
    # number of MQTT cameras published as Topic: MQTTcam/N, subscribed here as Topic: MQTTcam/#, Cams numbered 0 to N-1
    ap.add_argument("-Nmqtt", "--NmqttCams", type=int, default=0,
                    help="number of MQTT cameras published as Topic: MQTTcam/N,  Cams numbered 0 to N-1")
    # alternate, specify a list of camera numbers
    ap.add_argument("-camList", "--mqttCamList", type=int, nargs='+',
                    help="list of MQTTcam/N subscription topic numbers,  cam/N numbered from 0 to Nmqtt-1.")

    # specify display width and height
    ap.add_argument("-dw", "--displayWidth", type=int, default=1920, help="host display Width in pixels, default=1920")
    ap.add_argument("-dh", "--displayHeight", type=int, default=1080, help="host display Height in pixels, default=1080")

    # specify host display width and height of camera image
    ap.add_argument("-iw", "--imwinWidth", type=int, default=640, help="camera host display window Width in pixels, default=640")
    ap.add_argument("-ih", "--imwinHeight", type=int, default=360, help="camera host display window Height in pixels, default=360")

    # enable local save of detections on AI host, useful if node-red notification code is not being used
    ap.add_argument("-ls", "--localSave", action="store_true", help="save detection images on local AI host")
    # specify file path of location to same detection images on the localhost
    ap.add_argument("-sp", "--savePath", default="", help="path to location for saving detection images, default ~/detect")
    # save all processed images, fills disk quickly, really slows things down, but useful for test/debug
    ap.add_argument("-save", "--saveAll", action="store_true", help="save all images not just detections on host filesystem, for test/debug")

    # PiCamera module
    ap.add_argument("-pi", "--PiCam", action="store_true", help="Use Pi  camera module")

    args = vars(ap.parse_args())


    mqttCamsOneThread = args["mqttCamOneThread"]
    MQTTdemand = args["mqttDemand"]
    PiCAM = args["PiCam"]


# mark start of this code in log file
print("**************************************************************")
currentDT = datetime.datetime.now()
print("*** " + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
print("[INFO] using openCV-" + cv2.__version__)


# *** Function definitions
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# Boilerplate code to setup signal handler for graceful shutdown on Linux
def sigint_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        #print('caught SIGINT, normal exit. -- ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sighup_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGHUP! ** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigquit_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGQUIT! *** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
        QUIT=True

def sigterm_handler(signal, frame):
        global QUIT
        currentDT = datetime.datetime.now()
        print('caught SIGTERM! **** ' + currentDT.strftime("%Y-%m-%d  %H:%M:%S"))
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



# The callback for when a PUBLISH message is received from the server, aka message from SUBSCRIBE topic.
def on_message(client, userdata, msg):
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    global UImode
    global CameraToView
    if str(msg.topic) == "Alarm/MODE":          # Idle will not save detections, Audio & Notify are the same here
        currentDT = datetime.datetime.now()     # logfile entry
        AlarmMode = str(msg.payload.decode('utf-8'))
        print(str(msg.topic)+":  " + AlarmMode + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        return
    # UImode: 0->no Dasboard display, 1->live image from selected cameram 2->detections from selected camera, 3->detection from any camera
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


def on_publish(client, userdata, mid):
    #print("mid: " + str(mid))      # don't think I need to care about this for now, print for initial tests
    pass


def on_disconnect(client, userdata, rc):
    if rc != 0:
        currentDT = datetime.datetime.now()
        print("Unexpected MQTT disconnection!" + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S  "), clinet)
    pass


# callbacks for mqttCam that can't be shared
# mqttCamsOneThread=False is default
## True/False no significant difference on i7-6700K Desktop both ~53 fps for 15 ~5 fps MQTTcams from i5ai rtsp2mqtt server
               ## On Pi4, XU-4 etc.  one thread for all mqttCams is ~1.5 fps faster.
if not mqttCamsOneThread:   # use one mqtt thread per mqttCam
# callbacks for mqttCam that can't be shared
  def on_mqttCam_connect(client, userdata, flags, rc):
        camT=userdata[0]
        camN=userdata[1]
        client.subscribe("MQTTcam/"+str(camT), 0)


  def on_mqttCam(client, userdata, msg):
    global mqttCamOffset
    global inframe
    global mqttFrameDrops
    global mqttFrames
    # put input image into the camera's inframe queue
    try:
        camT=userdata[0]
        camN=userdata[1]
        mqttFrames[camN]+=1
        # thanks to @krambriw on the node-red user forum for clarifying this for me
        npimg=np.frombuffer(msg.payload, np.uint8)      # convert msg.payload to numpy array
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)   # decode image file into openCV image
        imageDT=datetime.datetime.now()
        if inframe[camN+mqttCamOffset].full():
            [_,_,_]=inframe[camN+mqttCamOffset].get(False)
            mqttFrameDrops[camN]+=1     # is happes here, shouldn't happen below
        inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset, imageDT), False)
        ##inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset), True, 0.200)
    except:
        mqttFrameDrops[camN]+=1     # queue.full() is not 100% reliable
    if MQTTdemand:
        client.publish(str("sendOne/" + str(camT)), "", 0, False)
##    time.sleep(0.001)     # force thread dispatch, hard to tell if this helps or not.
    return

else:
  def on_mqttCam_connect(client, camList, flags, rc):
     for camN in camList:
        client.subscribe("MQTTcam/"+str(camN), 0)


  def on_mqttCam(client, camList, msg):
    global mqttCamOffset
    global inframe
    global mqttFrameDrops
    global mqttFrames
    global Nmqtt    ## eliminate len(camList) call by using global
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
            imageDT=datetime.datetime.now()
            if inframe[camN+mqttCamOffset].full():
                [_,_,_]=inframe[camN+mqttCamOffset].get(False)
                mqttFrameDrops[camN]+=1     # is happes here, shouldn't happen below
            inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset, imageDT), False)
        except:
            mqttFrameDrops[camN]+=1     # queue.full() is not 100% reliable
        try:
            if MQTTdemand:
                client.publish(str("sendOne/" + str(camT)), "", 0, False)
##            time.sleep(0.001)     # force thread dispatch, hard to tell if this helps or not.
        except Exception as e:
            print("pub error " + str(e))
        return



# Hard to believe but Python threads don't have a terminate signal, need a kludge like this
# There are other ways, but I want some stats printed at thread termination.
# I think the real issue is the AI threads are in seperate Python files and its a scope issue for the Quit global
def QUITf():
    global QUIT
    return QUIT



# *** main()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main():
    global QUIT
    global AlarmMode    # would be Notify, Audio, or Idle, Idle mode doesn't save detections
    AlarmMode="Audio"   # will be Email, Audio, or Idle  via MQTT controller from alarmboneServer
    global CameraToView
    CameraToView=0
    global UImode
    UImode=0    # controls if MQTT buffers of processed images from selected camera are sent as topic: ImageBuffer
    global subscribeTopic
    global Nonvif
    global Nrtsp
    global Nmqtt
    global mqttCamOffset
    global mqttFrameDrops
    global inframe
    global Ncameras
    global mqttFrames
    global mqttCamsOneThread
    global __PYCORAL__

    # set variables from command line auguments or defaults
    nCoral = args["nTPU"]
    if nCoral > 1:
        nCoral = 1      # Not finished multiple TPU support, not sure it is needed or will be useful.
    nOVthreads = args["nNCS"]
    SSDv1 = args["SSDv1"]
    NCS_sdkV1 = args["sdkV1"]
    nCPUthreads = args["nAIcpuThreads"]
    useGPU = args["GPU"]
    confidence = args["confidence"]
    verifyConf = args["verifyConfidence"]
    noVerifyNeeded = args["noVerifyConfidence"]
    blobThreshold = args["blobFilter"]
    dbg=args["debug"]
    MQTTcameraServer = args["mqttCameraBroker"]
    Nmqtt = args["NmqttCams"]
    camList=args["mqttCamList"]
    if camList is not None:
        Nmqtt=len(camList)
    elif Nmqtt>0:
        camList=[]
    for i in range(Nmqtt):
        camList.append(i)
    dispMode = args["display"]
    if dispMode > 1:
        displayMode=1
    CAMERAS = args["cameraURLs"]
    RTSP = args["rtspURLs"]
    MQTTserver = args["mqttBroker"]     # this is for command and control messages, and detection messages
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
        print("[INFO] " + str(Nonvif) + " http Onvif snapshot threads will be created.")
    except:
        # No Onvif cameras
        print("[INFO] No " + str(CAMERAS) + " file.  No Onvif snapshot threads will be created.")
        Nonvif=0
    Ncameras=Nonvif


    # *** get rtsp URLs
    try:
        rtspURL=[line.rstrip() for line in open(RTSP)]
        Nrtsp=len(rtspURL)
        print("[INFO] " + str(Nrtsp) + " rtsp stream threads will be created.")
    except:
        # no rtsp cameras
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
    QDEPTH = 2      # bump up for trial of "read queue if full and then write to queue" in camera input thread
##    QDEPTH = 1      # small values improve latency
    print("[INFO] allocating camera and stream image queues...")
    if PiCAM:
        PiCamOffset=Ncameras
        Ncameras+=1
        print("[INFO] allocating queue for PiCamera Module...")
    mqttCamOffset = Ncameras
    mqttFrameDrops = 0
    mqttFrames = 0
    if Nmqtt > 0:
        print("[INFO] allocating " + str(Nmqtt) + " MQTT image queues...")
    Ncameras+=Nmqtt     # I generally expect Nmqtt to be zero if Ncameras is not zero at this point, but its not necessary
    if Ncameras == 0:
        print("[INFO] No Cameras, rtsp Streams, or MQTT image inputs specified!  Exiting...")
        quit()
    results = Queue(int(Ncameras/2)+2)
    inframe = list()
    for i in range(Ncameras):
        inframe.append(Queue(QDEPTH))


    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])


    # *** setup display windows if necessary
    # mostly for initial setup and testing, not worth a lot of effort at the moment
    if dispMode > 0:
        if Nonvif > 0:
            print("[INFO] setting up Onvif camera image windows ...")
            for i in range(Nonvif):
                name=str("Live_" + str(i))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                name=str("Live_" + str(i+Nonvif))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nmqtt > 0:
            print("[INFO] setting up MQTT camera image windows ...")
            for i in range(Nmqtt):
                name=str("Live_" + str(i+mqttCamOffset))
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.imshow(name, img)
                cv2.waitKey(1)



        # *** move windows into tiled grid
        top=20
        left=2
        ##left=1900           ## overrides for my 4K monitors
        ##displayHeight=1900  ## overrides for my 4K monitors
        Xshift=imwinWidth+3
        Yshift=imwinHeight+28
        Nrows=int(displayHeight/imwinHeight)
        for i in range(Ncameras):
            name=str("Live_" + str(i))
            col=int(i/Nrows)
            row=i%Nrows
            cv2.moveWindow(name, left+col*Xshift, top+row*Yshift)


    # *** connect to MQTT broker for control/status messages
    print("[INFO] connecting to MQTT " + MQTTserver + " broker...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.will_set("AI/Status", "Python AI has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()


    # this is for using legacy support for NSC v1 SDK code instead of OpenVINO
    SDKdevices = 0
    if NCS_sdkV1 and nOVthreads == 0:
        from mvnc import mvncapi as mvnc
        import NCS_sdkv1_Thread
        # grab a list of all NCS devices plugged in to USB
        print("[INFO] finding NCS SDK V1 devices...")
        devices = mvnc.EnumerateDevices()
        if len(devices) > 0:
            SDKdevices = len(devices)
            print("[INFO] found {} Movidius NCS devices.".format(SDKdevices))
            # open the CNN graph file
            print("       loading the graph file into memory...")
            with open("./graphs/mobilenetgraph", mode="rb") as f:
                graph_in_memory = f.read()
            device = list()
            graph = list()
            for devnum in range(SDKdevices):
                print("       opening device{} ...".format(devnum))
                device.append(mvnc.Device(devices[devnum]))
                device[devnum].OpenDevice()
                print("       allocating graph on NCS device{} ...".format(devnum))
                graph.append(device[devnum].AllocateGraph(graph_in_memory))
            print("[INFO] starting " + str(SDKdevices) + " Movidius NCS SDK V1 AI Threads ...")
            AIt = list()
            for i in range(SDKdevices):
                AIt.append(Thread(target=NCS_sdkv1_Thread.AI_thread,
                    args=(results, inframe, graph[i], i, cameraLock, nextCamera, Ncameras,
                        PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, QUITf, blobThreshold)))
                AIt[i].start()

    if SDKdevices+nCoral+nOVthreads+nCPUthreads == 0:
        print("[INFO] No Coral TPU or Myriad NCS/NCS2 devices specified, forcing one CPU AI thread.")
        nCPUthreads=1   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+


    # *** setup and start Coral AI threads
    # Might consider moving this into the thread function.
    ### Setup Coral AI
    # initialize the labels dictionary
    if nCoral > 0:
        import Coral_TPU_Thread
        print("[INFO] parsing mobilenet_ssd_v2 coco class labels for Coral TPU...")
        if Coral_TPU_Thread.__PYCORAL__ is False:
            labels = {}
            for row in open("mobilenet_ssd_v2/coco_labels.txt"):
                # unpack the row and update the labels dictionary
                (classID, label) = row.strip().split(maxsplit=1)
                labels[int(classID)] = label.strip()
            print("[INFO] loading Coral mobilenet_ssd_v2_coco model...")
            model = Coral_TPU_Thread.DetectionEngine("mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
        else:
            labels = Coral_TPU_Thread.read_label_file("mobilenet_ssd_v2/coco_labels.txt")
            model = Coral_TPU_Thread.make_interpreter("mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
            model.allocate_tensors()

        # *** start Coral TPU threads
        Ct = list() ## not necessary only supporting a single TPU for now.
        print("[INFO] starting " + str(nCoral) + " Coral TPU AI Threads ...")
        for i in range(nCoral):
            print("... loading model...")
            Ct.append(Thread(target=Coral_TPU_Thread.AI_thread,
                args=(results, inframe, model, labels, i, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, QUITf, blobThreshold)))
            Ct[i].start()


    # *** setup and start Myriad OpenVINO
    ## Hmmm... single NCS, Caffe SSDv1 ~9.7 fps with 5 Onvif cameras,  TensorFlow SSDv2  gets only ~5.7 fps, with NCS2 ~11.8 fps
    if nOVthreads > 0:
      if cv2.__version__.find("openvino") > 0:
        import OpenVINO_Thread
        if SSDv1:
            print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO Myriad NCS/NCS2 AI threads...")
            OVstr = "CaffeSSD"
        else:
            ## fragile works for 2021.1, need better way to detect openVINO version lacks NCS support and needs IR10 models
            if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino":
                print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 IR10 model for OpenVINO_2021.1 Myriad NCS2 AI threads...")
                OVstr = "SSDv2_IR10"
            else:
                print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 model for OpenVINO Myriad NCS/NCS2 AI threads...")
                OVstr = "SSDv2ncs"
        netOV=list()
        for i in range(nOVthreads):
            print("... loading model...")
            if SSDv1:
                netOV.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
            else:
                ## fragile works for 2021.1, need better way to detect openVINO version lacks NCS support and needs IR10 models
                if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino":
                    netOV.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.bin"))
                else:
                    netOV.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2coco.xml", "mobilenet_ssd_v2/MobilenetSSDv2coco.bin"))
            netOV[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            netOV[i].setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)  # specify the target device as the Myriad processor on the NCS
        # *** start OpenVINO AI threads
        OVt = list()
        print("[INFO] starting " + str(nOVthreads) + " OpenVINO Myriad NCS/NCS2 AI Threads ...")
        for i in range(nOVthreads):
            OVt.append(Thread(target=OpenVINO_Thread.AI_thread,
                args=(results, inframe, netOV[i], i, cameraLock, nextCamera, Ncameras,
                    PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, OVstr, QUITf, blobThreshold, SSDv1)))
            OVt[i].start()
      else:
        print("[ERROR!] OpenVINO version of openCV is not active, check $PYTHONPATH")
        print(" No MYRIAD (NCS/NCS2) OpenVINO threads will be created!")
        nOVthreads = 0
        if nCoral+nCPUthreads == 0:
            print("[INFO] No Coral TPU device or CPU threads specified, forcing one CPU AI thread.")
            nCPUthreads=1   # we always can force one CPU thread, but ~1.8 seconds/frame on Pi3B+


    # ** setup and start CPU AI threads, usually only one makes sense.
    ## TODO: do I want SSDv2 option for CPU threads as well??  Done, Made FP32 version with Model Optimizer.
    ## Will need to make FP32 version,  SSDv2 error: "Inference Engine backend: The plugin does not support FP16 in function 'initPlugin'"
    if nCPUthreads > 0:
        net=list()
        if cv2.__version__.find("openvino") > 0:
            import OpenVINO_Thread
            if SSDv1:
                print("[INFO] loading Caffe Mobilenet-SSD model for OpenVINO CPU AI threads...")
                OVstr = "SSDv1_cpu"
                for i in range(nCPUthreads):
                    net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
                    net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                    net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino":
                    print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP16 IR10 model for OpenVINO_2021.1...")
                    if useGPU:
                        OVstr = "SSDv2_IR10gpu"
                    else:
                        OVstr = "SSDv2_IR10cpu"
                else:
                    print("[INFO] loading Tensor Flow Mobilenet-SSD v2 FP32 model for OpenVINO CPU AI threads...")
                    OVstr = "SSDv2cpu"
                for i in range(nCPUthreads):
                    if cv2.__version__ == "4.5.0-openvino" or cv2.__version__ == "4.5.1-openvino":
                        net.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoIR10.bin"))
                        net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                        if useGPU:
                            print("Using OPEN_CL_FP16 GPU instead of CPU")
                            net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                        else:
                            net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    else:
                        net.append(cv2.dnn.readNet("mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.xml", "mobilenet_ssd_v2/MobilenetSSDv2cocoFP32.bin"))
                        net[i].setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                        net[i].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            import ocvdnn_CPU_Thread
            print("[INFO] loading Caffe Mobilenet-SSD model for ocvdnn CPU AI threads...")
            for i in range(nCPUthreads):
                net.append(cv2.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel"))
        # *** start CPU AI threads
        CPUt = list()
        if cv2.__version__.find("openvino") > 0:
            if useGPU:
                print("[INFO] starting " + str(nCPUthreads) + " OpenVINO GPU AI Threads ...")
            else:
                print("[INFO] starting " + str(nCPUthreads) + " OpenVINO CPU AI Threads ...")
        else:
            print("[INFO] starting " + str(nCPUthreads) + " openCV dnn module CPU AI Threads ...")
        for i in range(nCPUthreads):
            if cv2.__version__.find("openvino") > 0:
                CPUt.append(Thread(target=OpenVINO_Thread.AI_thread,
                    args=(results, inframe, net[i], i, cameraLock, nextCamera, Ncameras,
                        PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, OVstr, QUITf, blobThreshold, SSDv1)))
            else:
                CPUt.append(Thread(target=ocvdnn_CPU_Thread.AI_thread,
                    args=(results, inframe, net[i], i, cameraLock, nextCamera, Ncameras,
                        PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf, dbg, QUITf, blobThreshold)))
            CPUt[i].start()




    # *** Open second MQTT client thread for MQTTcam/# messages "MQTT cameras"
    # Requires rtsp2mqttDemand.py mqtt camera source
    # mqttCamsOneThread lets me try one mqtt thread for all MQTT cameras, need to re-evaluate after recent change to rtsp2mqttPdemand.py
    if Nmqtt > 0:
      mqttFrameDrops=[]
      mqttFrames=[]
      mqttCam=list()
      print("[INFO] connecting to " + MQTTcameraServer + " broker for MQTT cameras...")
      if not mqttCamsOneThread:   # use one MQTT thread per camera
        print("INFO starting one thread per MQTT camera.")
        j=0
        for i in camList:
            mqttFrameDrops.append(0)
            mqttFrames.append(0)
            mqttCam.append(mqtt.Client(userdata=(i, j), clean_session=True))
            mqttCam[j].on_connect = on_mqttCam_connect
            mqttCam[j].on_message = on_mqttCam
            mqttCam[j].on_publish = on_publish
            mqttCam[j].on_disconnect = on_disconnect
            mqttCam[j].connect(MQTTcameraServer, 1883, 60)
            mqttCam[j].loop_start()
            time.sleep(0.1)     # force thread dispatch
            if MQTTdemand:
                mqttCam[j].publish(str("sendOne/" + str(i)), "", 0, False)   # start messages
            j+=1
      else: # one MQTT thread for all cameras
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
        if MQTTdemand:
            for i in camList:
                mqttCam.publish(str("sendOne/" + str(i)), "", 0, False)   # start messages


    # *** start camera reading threads
    o = list()
    if Nonvif > 0:
        import onvif_Thread
        print("[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        for i in range(Nonvif):
            o.append(Thread(target=onvif_Thread.onvif_thread, args=(inframe[i], i, CameraURL[i], QUITf)))
            o[i].start()

    if PiCAM:
        PiCAM_DIMS = (1296, 976)      # 1296x972 is suposed to be "more efficient" in picamara docs but imutils needs divisible by 8 values
        print("[INFO] starting Pi Camera Module Thread ...")
        Pi_vs = PiVideoStream(inframe[PiCamOffset], PiCamOffset, resolution=PiCAM_DIMS).start()
        time.sleep(2)

    if Nrtsp > 0:
        global threadLock
        global threadsRunning
        threadLock = Lock()
        threadsRunning = 0
        ###import rtsp_Thread
        print("[INFO] starting " + str(Nrtsp) + " RTSP Camera Sampling Threads ...")
        for i in range(Nrtsp):
            ##o.append(Thread(target=rtsp_Thread.rtsp_thread, args=(inframe[i+Nonvif], i, rtspURL[i], QUITf)))
            o.append(Thread(target=rtsp_thread, args=(inframe[i+Nonvif], i+Nonvif, rtspURL[i], QUITf)))
            o[i+Nonvif].start()
        while threadsRunning < Nrtsp:
            time.sleep(0.5)
        print("[INFO] All " + str(Nrtsp) + " RTSP Camera Sampling Threads are running.")



    #*************************************************************************************************************************************
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    SEND_ALIVE=100  # send MQTT message approx. every SEND_ALIVE/fps seconds to reset external "watchdog" timer for auto reboot.
    waitCnt=0
    prevUImode=UImode
    currentDT = datetime.datetime.now()
    print("[INFO] AI/Status: Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.publish("AI/Status", "Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    # *** MQTT send a blank image to the dashboard UI
    print("[INFO] Clearing dashboard ...")
    client.publish("ImageBuffer/!AI has Started.", bytearray(img_as_jpg), 0, False)
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
                aliveCount = (aliveCount+1) % SEND_ALIVE   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                continue
            if img is not None:
                fps.update()    # update the FPS counter
                # setup for file saving
                folder=dt.strftime("%Y-%m-%d")
                filename=dt.strftime("%H_%M_%S.%f")
                filename=filename[:-5] + "_" + ai  #just keep tenths, append AI source
                if localSave:
                    lfolder=str(detectPath + "/" + folder)
                    if os.path.exists(lfolder) == False:
                        os.mkdir(lfolder)
                    if personDetected:
                        outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                    else:   # in case saveAll option
                        outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                    if (personDetected and not AlarmMode.count("Idle")) or saveAll:  # save detected image
                        cv2.imwrite(outName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if personDetected:
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    if retv:
                        outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                        outName=outName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                        client.publish(str(outName), bytearray(img_as_jpg), 0, False)
##                        print(outName)  # log detections
                    else:
                        print("[INFO] conversion of np array to jpg in buffer failed!")
                        continue
                # send image for live display in dashboard
                if ((CameraToView == cami) and (UImode == 1 or (UImode == 2 and personDetected))) or (UImode ==3 and personDetected):
                    if personDetected:
                        topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                    else:
                        retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                        if retv:
                            topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +".jpg")
                        else:
                            print("[INFO] conversion of numpy array to jpg in buffer failed!")
                            continue
                    client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                # display the frame to the screen if enabled, in normal usage display is 0 (off)
                if dispMode > 0:
                    name=str("Live_" + str(cami))
                    cv2.imshow(name, cv2.resize(img, (imwinWidth, imwinHeight)))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"): # if the `q` key was pressed, break from the loop
                        QUIT=True   # exit main loop
                        continue
                aliveCount = (aliveCount+1) % SEND_ALIVE
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                if prevUImode != UImode:
                    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
                    img[:,:] = (154,127,100)
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    client.publish("ImageBuffer/!AI Mode Changed.", bytearray(img_as_jpg), 0, False)
                    prevUImode=UImode
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
    #*************************************************************************************************************************************



    # *** Clean up for program exit
    fps.stop()    # stop the FPS counter timer
    currentDT = datetime.datetime.now()
    print("[INFO] Program Exit signal received:" + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    # display FPS information
    print("*** AI processing approx. FPS: {:.2f} ***".format(fps.fps()))
    print("[INFO] Run elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Frames processed by AI system: " + str(fps._numFrames))
    print("[INFO] Main loop waited for results: " + str(waitCnt) + " times.")
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    print("[INFO] AI/Status: Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))


    # stop cameras
    if Nmqtt > 0:
      if not mqttCamsOneThread:
        for i in range(Nmqtt):
            mqttCam[i].disconnect()
            mqttCam[i].loop_stop()
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))
      else:
        mqttCam.disconnect()
        mqttCam.loop_stop()
        for i in range(Nmqtt):
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))
    if PiCAM:
        Pi_vs.stop()

    # wait for threads to exit
    if Nonvif > 0:
        for i in range(Nonvif):
            o[i].join()
        print("[INFO] All Onvif Camera have exited ...")
    if Nrtsp > 0:
        for i in range(Nrtsp):
            o[i+Nonvif].join()
        print("[INFO] All rtsp Camera have exited ...")

    # wait for threads to exit
    if nCPUthreads > 0:
        for i in range(nCPUthreads):
            CPUt[i].join()
        print("[INFO] All CPU AI Threads have exited ...")

    if nOVthreads > 0:
        for i in range(nOVthreads):
            OVt[i].join()
        print("[INFO] All OpenVINO Myriad NCS/NCS2 AI Threads have exited ...")

    if nCoral > 0:
        for i in range(nCoral):
            Ct[i].join()
        print("[INFO] All Coral TPU AI Threads have exited ...")

    if SDKdevices > 0:
        for i in range(SDKdevices):
            AIt[i].join()
        print("[INFO] All NCS SDK V1 AI Threads have exited ...")


    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()



    # Send a blank image the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros((imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (32,32,32)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    client.publish("ImageBuffer/!AI has Exited.", bytearray(img_as_jpg), 0, False)
    time.sleep(1.0)



    # clean up MQTT
    client.disconnect()     # normal exit, Will message should not be sent.
    currentDT = datetime.datetime.now()
    print("[INFO] Stopping MQTT Threads." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.loop_stop()      ### Stop MQTT thread

    # clean up the NCS graph and device
    if SDKdevices > 0:
        for devnum in range(SDKdevices):
            graph[devnum].DeallocateGraph()
            device[devnum].CloseDevice()


    # bye-bye
    currentDT = datetime.datetime.now()
    print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    print("")
    print("")



# *** RTSP Sampling Thread
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def rtsp_thread(inframe, camn, URL, QUITf):
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
    while not QUITf():
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
                time.sleep(30.0)
                Rcap=cv2.VideoCapture(URL)
                Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not Rcap.isOpened():
                    if not Error2:
                        Error2=True
                        currentDT = datetime.datetime.now()
                        print('[Error2!] RTSP stream'+ str(camn) + ' re-open failed! $$$ ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                        print('*** Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(30.0)    # takes ~4 minutes to recover when Lorex auto reboots.
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
            time.sleep(10.0)    # So far, I've never seen this Exception message.
        try:
            if frame is not None:
                imageDT=datetime.datetime.now()
                if inframe.full():
                    [_,_,_]=inframe.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below
                inframe.put((frame, camn, imageDT), False)   # no block if queue full, go grab fresher frame
        except: # most likely queue is full, Python queue.full() is not 100% reliable
            # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
            ocnt+=1
    Rcap.release()
    print("RTSP stream sampling thread" + str(camn) + " is exiting, dropped frames " + str(ocnt) + " times.")



# *** PiCamera Module support
#***************************************************************************************************************************
if PiCAM:
  from picamera.array import PiRGBArray
  from picamera import PiCamera

# modified VideoStream class from imutils library, I find the PiCamera is not really suitable for 24/7
# usage.  These modifications recover from some of the failures, but occasionally still need to reboot
# or in somewhat rare cases, cycle the power.  Could be power releated,since putting test system on UPS
# its ran for over 10 weeks withonly a single error, from which it automatically recovered.

  class PiVideoStream:
    def __init__(self, inQueue, camn, resolution=(320, 240), framerate=16):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        self.inQueue = inQueue
        self.camn = camn
        # initialize the frame and the variable used to indicate if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.error=False
        self.ocnt=0
        self.imageDT=datetime.datetime.now()

    def start(self):
        # start the thread to read frames from the video stream
        print("[INFO] Pi Camera Module Thread is running, resolution:", self.camera.resolution)
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        try:
            for f in self.stream:
                # grab the frame from the stream and clear the stream in preparation for the next frame
                self.frame = f.array
                self.rawCapture.truncate(0)
                self.error = False
                self.imageDT = datetime.datetime.now()
                try:
                  if self.inQueue.full():
                    [_,_,_]=self.inQueue.get(False)    # remove oldest sample to make space in queue
                    self.ocnt+=1     # if happens here shouldn't happen below
                  self.inQueue.put((self.frame, self.camn, self.imageDT), False)   # no block if queue full, go grab fresher frame
                except:
                    self.ocnt+=1
        except Exception as e:
            self.currentDT = datetime.datetime.now()
            ##if self.error is False:
            print(" **** PiCamera error: " + str(e)  + self.currentDT.strftime(" -- %Y-%m-%d_%H_%M_%S.%f"))
            self.error=True
            # attempt to recover
            try:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                time.sleep(2)
                self.camera = PiCamera()
                #self.camera.resolution = DISPLAY_DIMS
                #self.camera.framerate = 32
                self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
                self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
                time.sleep(2)
                self.frame = None
                self.update()   # trapping the error but nothing else seems to happen need to re-enter for loop?
            except Exception as e:
                print(" @@@ PiCamera recovery error: " + str(e)  + self.currentDT.strftime(" -- %Y-%m-%d_%H_%M_%S.%f"))

                # if the thread indicator variable is set, stop the thread and resource camera resources
                if self.stopped:
                    self.stream.close()
                    self.rawCapture.close()
                    self.camera.close()
                    return

    def read(self):
	  # return the frame most recently read
      if self.error is False:
        return self.frame
      else:
            return None

    def stop(self):
      # indicate that the thread should be stopped
      self.stopped = True





# python boilerplate
if __name__ == '__main__':
    main()


