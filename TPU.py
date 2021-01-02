#!/usr/bin/env python3
#
### 26JUL2020wbk
# Having stumbled onto some easy to use code for fisheye camera "de-warping" it was pretty straghtforward
# to modify the virtual camera thread to de-warp and queue multiple virtual PTZ views from a single fisheye camera.
# As with Virtual cameras, Fisheye setup is done by editing the code as there are too many parameters for a "simple"
# configuration file to work.
# Regular camera images can be used to test the code, Roll Pitch Yaw movements don't work as expected but the process
# can be debugged.
#
#
###
# 11MAY2020wbk
# Spin-off version to crop regions from a high resolution camera and treat them as seperate cameras.
# For now this needs to be edited to setup the virtual cameras, not sure its generally useful enough
# to bother with passing all the required additional settings.
#
#
### MendelTPU.py 16AUG2019wbk
# Note, Coral Developemnt Board mosquitto/systemd bug doesn't start after bootup, do if using localhost mqtt: sudo mosquitto &
# quick fix add to root crontab:
#   @reboot /usr/sbin/mosquitto >/dev/null 2>&1 &
#
#
# Derived from AI_dev.py
# This is basically AI_dev with everything removed except for the TPU AI thread, Onvif, rtsp, and mqttCam threads.  All code
# is moved back into this single python file.  Not sure it ends up any different from TPU.py except for some Coral development
# board "Mendel" ID strings, and the local saving of detection images being removed.
#
## First steps to making AI person detection "appliance".  Run the AI and send detections to central MQTT broker.
#
# ~12.4 fps on Coral Mendel Dev board with 6 i5ai mqttCams and -d 1 live image display.
# ~14.8 fps witih -d 0 no display option.
#
## 17AUG2019wbk
# Add detection box points (startX, startY, endX, endY) as part of MQTT Topic string for possible post processing detection filter.
# Reorginize main loop to avoid imwrite() and imencode() if results are not going to be used.
#
## 23AUG2019wbk
# Performance test on i7-6700K Desktop: ./MendelTPU.py -camMQTT i5ai -Nmqtt 15 -mqtt localhost -d 1 --> ~43.7 fps.
# Note that ~45 fps is processing every frame from all 15 Lorex DVR rtsp streams, ~18 fps would be every frame from the camList.
# Corel TPU Developement Board: ./MendelTPU.py -camList 1 2 3 5 6 14 --> ~15.5 fps for a run of ~1.5 day.
# Pi4B with Coral TPU USB3 stick: ./MendelTPU.py -camList 1 2 3 5 6 14 -sys Pi4B  --> ~16.0 fps for ~2 hr run.
# Odroid XU-4 with TPU USB3 stick: ./MendelTPU.py -camList 1 2 3 5 6 14 -mqtt kahuna --> 11.5 fps for ~2.5 hr run.
#
# 25AUG2019wbk
# Nvidia Jetson Nano:
#   ./MendelTPU.py -camList 1 2 3 5 6 14 -mqtt kahuna.local -sys Jetson -camMQTT i5ai.local  --> 17.7 fps, effectively every frame!
#   ./MendelTPU.py -Nmqtt 15 -mqtt kahuna.local -sys Jetson -camMQTT i5ai.local --> ~24.8 fps.
#   ./MendelTPU.py -camList 0 1 2 3 4 5 6 8 9 10 14 -mqtt kahuna.local -sys Jetson -camMQTT i5ai.local --> ~25.7 fps for ~24 Hr run.
# NOTE: the above performace tests are with 1080p HD camera streams.  The rtsp2mqtt.py "server" and "mqtt cams" looked like a
# good solution.  Unfortunately it didn't scale well at all when I upgraded to 4K UHD cameras
#
# 17OCT2019wbk -- Add syncronized wait to rtsp thread startup, improves fps measurement for short runs.
#
# 5DEC2019wbk some Pi4B tests with rtsp cameras, 3fps per stream:
# 4 UHD (4K)  :     ~2.8 fps
# 4 HD (1080p):     ~11.8 fps (basically processing every frame)
# 2 UHD 2 HD  :     ~6.7 fps (Pi4B struggles with 4K streams)
# 5 HD        :     ~14.7 fps (basically processing every frame)
# 6 HD        :     ~15.0 fps, -d 0 (no display) ~16.7 fps
# 8 HD        :    ~11.6 fps, -d 0 ~14.6 fps
#
## 6DEC2019wbk Some UHD tests on Jetson Nano
# 5 UHD (4K)         :  ~14.6 fps (effectively processing every frame!)
# 5 UHD 3 HD         :  ~10.3 fps, jumps to ~19.1 fps if -d 0 option used (no live image display)
# 4 UHD 4 HD         :  ~16.3 fps, ~22.5 fps with -d 0 option
# 5 UHD 10 HD (1080p):  ~4.4 fps, ~7.6 fps with -d 0 option (totally overloaded, get ~39 fps with running on i7-4500U MiniPC)
#
## 7DEC2019wbk Coral Development Board
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


# import the necessary packages
import platform
global __WIN__
if platform.system()[:3] != 'Lin':
    __WIN__ = True
else:
    __WIN__ = False
    import signal

import sys
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
from threading import Thread, Lock

# TPU
from edgetpu.detection.engine import DetectionEngine
from edgetpu import __version__ as edgetpu_version

# for saving PTZ view maps
import pickle


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
global dbg
global CamName


## Specific for my use
# Map Lorex camera names to camera numbers, Lorex uses 1-16, Python uses 0-15
## After Lorex died, I reorderd the cameras on the Qcamera DVR-16 replacement for "better" mosaic display
LorexName = [
    "MailBox",
    "HummingbirdLeft",      # 4K
    "LEFT",
    "RIGHT",
    "FrontDoor",            # 4K
    "HummingbirdRight",     # 4K
    "Intersection",
    "Cliffwood",            # 4K
    "DriveWay",
    "Shed",
    "KitchenDoor",
    "Garage",
    "Patio",
    "SideYard",
    "PoolEquipment",
    "PoolDeck",             # 4K
    "PoolShallowEnd",
    "Cam16"
]

CamName=[
    "Cam0",
    "Cam1",
    "Cam2",
    "Cam3",
    "Cam4",
    "Cam5",
    "Cam6",
    "Cam7",
    "Cam8",
    "Cam9",
    "Cam10",
    "Cam11",
    "Cam12",
    "Cam13",
    "Cam14",
    "Cam15"
]


# *** constants for MobileNet-SSD & MobileNet-SSD_V2  AI models
# frame dimensions should be square for MobileNet-SSD
PREPROCESS_DIMS = (300, 300)


# mark start of this code in log file
print("**************************************************************")
currentDT = datetime.datetime.now()
print("*** " + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
print("[INFO] using openCV-" + cv2.__version__)
print('Edgetpu_api version: ' + edgetpu_version)


# *** Function definitions
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# Boilerplate code to setup signal handler for graceful shutdown on Linux
if __WIN__ is False:
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
### The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    global subscribeTopic
    #print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.  -- straight from Paho-Mqtt docs!
    client.subscribe(subscribeTopic)


###*******************************************************************************************************
#   With saving of detections moved to the -mqtt "controller" host, we always stay in "Audio" mode and let the
# controller decide to save detections or not based on the Alarm mode.  Easier to do this in Node-red,
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
            dt=datetime.datetime.now()
            # thanks to @krambriw on the node-red user forum for clarifying this for me
            npimg=np.frombuffer(msg.payload, np.uint8)      # convert msg.payload to numpy array
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)   # decode image file into openCV image
            inframe[camN+mqttCamOffset].put((frame, camN+mqttCamOffset, dt), False)
        except:
            mqttFrameDrops[camN]+=1
        try:
            client.publish(str("sendOne/" + str(camT)), "", 0, False)
        except Exception as e:
            print("pub error " + str(e))
        return




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
    global CamName
    global blobThreshold


    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()

    ap.add_argument("-c", "--confidence", type=float, default=0.60, help="detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=0.70, help="detection confidence for verification")
    ap.add_argument("-nvc", "--noVerifyConfidence", type=float, default=.98, help="initial detection confidence to skip verification")
    ap.add_argument("-blob", "--blobFilter", type=float, default=.20, help="reject detections that are more than this fraction of the frame")

    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="MYcameraURL.rtsp", help="path to file containing rtsp camera stream URLs")

    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="MYcameraURL.txt", help="path to file containing http camera jpeg image URLs")

    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", type=int, default=1,
        help="display images on host screen, 0=no display, 1=live display")

    # specify MQTT broker
    ap.add_argument("-mqtt", "--mqttBroker", default="localhost", help="name or IP of MQTT Broker for control and detection storage.")

    # specify MQTT broker for camera images via MQTT, if not "localhost"
    ap.add_argument("-camMQTT", "--mqttCameraBroker", default="localhost", help="name or IP of MQTTcam/# message broker")
    # number of MQTT cameras published as Topic: MQTTcam/N, subscribed here as Topic: MQTTcam/#, Cams numbered 0 to N-1
    ap.add_argument("-Nmqtt", "--NmqttCams", type=int, default=0, help="number of MQTT cameras published as Topic: MQTTcam/N,  Cams numbered 0 to N-1")
    # alternate, specify a list of camera numbers
    ap.add_argument("-camList", "--mqttCamList", type=int, nargs='+', help="list of MQTTcam/N subscription topic numbers,  cam/N numbered from 0 to Nmqtt-1.")

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
    MQTTcameraServer = args["mqttCameraBroker"]
    Nmqtt = args["NmqttCams"]
    camList=args["mqttCamList"]
    if camList is not None:
        Nmqtt=len(camList)
        for i in camList:
            CamName.append(LorexName[i])
    elif Nmqtt>0:
        camList=[]
        for i in range(Nmqtt):
            camList.append(i)
            CamName.append(LorexName[i])
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
        CameraURL=[line.rstrip() for line in open(CAMERAS)]
        Nonvif=len(CameraURL)
        print("[INFO] " + str(Nonvif) + " http Onvif snapshot threads will be created.")
    except:
        # No Onvif cameras
        print("[INFO] No " + str(CAMERAS) + " file.  No Onvif snapshot threads will be created.")
        Nonvif=0
    Ncameras=Nonvif


    # *** get rtsp URLs
    # expecting rtsp stream URLs but things like /dev/video0 for a USB webcam will work, as will MJPEG URLs from motioneye OS etc.
    try:
        rtspURL=[line.rstrip() for line in open(RTSP)]
        Nrtsp=len(rtspURL)
        print("[INFO] " + str(Nrtsp) + " rtsp stream threads will be created.")
    except:
        # no rtsp cameras
        print("[INFO] No " + str(RTSP) + " file.  No rtsp stream threads will be created.")
        Nrtsp=0
    Ncameras+=Nrtsp


    # define virtual cameras, not generally useful, created for a specific test case, leave code inactive
    # if some reason turning one camera into multiple virtual cameras via image crops becomes useful again
    # For example MobilenetSSD_V1 performed poorly with cameras greater than 1080p resolution.
    if False:
        print("[INFO] Setting up virtual cameras with crops from hires cameras ...")
        VIRTrtspURL = [
            "rtsp://admin:aiVision77@reolink:554//h264Preview_01_main",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=2&stream=0.sdp"
        ]
        Nvirt=len(VIRTrtspURL)     # modified rtsp thread will crop sub-frames into multiple camera queues
        VIRTcrop = [
            #[ [ULx,ULy,LRx,LRy], [ULx,ULy,LRx,LRy], ... ]
            # I set the crop boxes by opening a full frame image in GIMP setting selection tool to fixed aspect ratio
            # of 16:9 or 4:3 and using the pixel cooridnates in shown in the selection dool settings dialog.
            [ [0,0, 1280,720], [1280,0, 2560,720] ],
            [ [0,0, 960,620], [960,0, 1920,620] ]
        ]
        NvirtCam=0
        for i in range(Nvirt):
            NvirtCam += len(VIRTcrop[i])
    else:
        NvirtCam=0
        Nvirt=0
    VirtCamOffset=Ncameras
    Ncameras+=NvirtCam # add fake cameras to count


    # define fisheye cameras and virtual PTZ views
    # fisheye.rtsp is expected to be created with the interactive fisheye_window C++ utility program
    try:
        l=[line.rstrip() for line in open('fisheye.rtsp')]
        FErtspURL=list()
        PTZparam=list()
        j=-1
        for i in range(len(l)):
            if not l[i]: continue
            if l[i].startswith('rtsp'):
                FErtspURL.append(l[i])
                j+=1
                PTZparam.append([])
            else:
                PTZparam[j].append(l[i].strip().split(' '))

        print("[INFO] Setting up PTZ virtual cameras views from fisheye camera ...")
        #print(FErtspURL)
        #print(PTZparam)
        Nfisheye=len(FErtspURL)     # modified rtsp thread will send PTZ views to seperate queues, this is number of fisheye threads
        NfeCam=0                    # total number of queues to be created for virtual PTZ cameras
        for i in range(Nfisheye):
            if len(PTZparam[i])<2 or len(PTZparam[i][0])<2 or len(PTZparam[i][1])!=6:
                # this is where Python's features make code simple but obtuse!
                # setting up this data structure in C/C++ gives me cooties with the variable number of possible PTZ views per camera!
                print('[ERROR] PTZparam[' + str(i) + '] must contain [srcW, srcH],[dstW,detH,  alpha,beta,theta,zoom] entries, Exiting ...')
                quit()
            NfeCam += len(PTZparam[i])-1 # the first entry is camera resolution, not a PTZ view
    except:
        # no fisheye cameras
        print("[INFO] No fisheye.rtsp file.  No fisheye camera rtsp stream threads will be created.")
        NfeCam=0
        Nfisheye=0
    FishEyeOffset=Ncameras
    Ncameras+=NfeCam # add PTZ views to cameras count


    # *** setup path to save AI detection images
    if savePath == "":
        detectPath= os.getcwd()
        if __WIN__ is False:
            detectPath=detectPath + "/detect"
        else:
            detectPath=detectPath + "\\detect"
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
    mqttCamOffset = Ncameras
    mqttFrameDrops = 0
    mqttFrames = 0
    Ncameras+=Nmqtt     # I generally expect Nmqtt to be zero if Ncameras is not zero at this point, but its not necessary
    if Ncameras == 0:
        print("[INFO] No Cameras, rtsp Streams, or MQTT image inputs specified!  Exiting...")
        quit()
    if Nmqtt > 0:
        print("[INFO] allocating " + str(Nmqtt) + " MQTT image queues...")
##    results = Queue(2*Ncameras)
    results = Queue(int(Ncameras/2)+1)
    inframe = list()
    for i in range(Ncameras):
        inframe.append(Queue(QDEPTH))



    # *** setup display windows if necessary
    # mostly for initial setup and testing, not worth a lot of effort at the moment
    if dispMode > 0:
        if Nonvif > 0:
            print("[INFO] setting up Onvif camera image windows ...")
            for i in range(Nonvif):
                name=CamName[i]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                name=CamName[i+Nonvif]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if NvirtCam > 0:
            print("[INFO] setting up  virtual camera image windows ...")
            for i in range(NvirtCam):
                name=CamName[i+VirtCamOffset]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if NfeCam > 0:
            print("[INFO] setting up  FishEye camera PTZ windows ...")
            for i in range(NfeCam):
                name=CamName[i+FishEyeOffset]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nmqtt > 0:
            print("[INFO] setting up MQTT camera image windows ...")
            for i in range(Nmqtt):
                name=CamName[i+mqttCamOffset]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)

        # *** move windows into tiled grid
        top=2
        left=1
        Xshift=imwinWidth+3
        Yshift=imwinHeight+28
        Nrows=int(displayHeight/imwinHeight)
        for i in range(Ncameras):
            #name=str("Live_" + str(i))
            name=CamName[i]
            col=int(i/Nrows)
            row=i%Nrows
            cv2.moveWindow(name, left+col*Xshift, top+row*Yshift)


    # *** connect to MQTT broker for control/status messages
    print("[INFO] connecting to MQTT " + MQTTserver + " broker for control and AI detection results...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    client.will_set("AI/Status:  Python AI has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()

    # *** MQTT send a blank image to the dashboard UI
    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    print("[INFO] Clearing dashboard ...")
    client.publish("ImageBuffer/!AI has Started.", bytearray(img_as_jpg), 0, False)


    # *** setup and start Coral AI threads
    # Might consider moving this into the thread function.
    ### Setup Coral AI
    # initialize the labels dictionary
    print("[INFO] parsing mobilenet_ssd_v2 coco class labels for Coral TPU...")
    labels = {}
    for row in open("mobilenet_ssd_v2/coco_labels.txt"):
        # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
    print("[INFO] loading Coral mobilenet_ssd_v2_coco model...")
    model = DetectionEngine("mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")



    # *** Open second MQTT client thread for MQTTcam/# messages for "MQTT cameras"
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
        for i in camList:
            mqttCam.publish(str("sendOne/" + str(i)), "", 0, False)   # start messages flowing



    # *** start camera reading threads
    o = list()
    if Nonvif > 0:
        print("[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        for i in range(Nonvif):
            o.append(Thread(target=onvif_thread, args=(inframe[i], i, CameraURL[i])))
            o[i].start()
    if Nrtsp+Nvirt+Nfisheye > 0:
        global threadLock
        global threadsRunning
        threadLock = Lock()
        threadsRunning = 0
        for i in range(Nrtsp):
            o.append(Thread(target=rtsp_thread, args=(inframe[i+Nonvif], i, rtspURL[i])))
            o[i+Nonvif].start()
        VCoffset=VirtCamOffset
        for i in range(Nvirt):
            Nvc=len(VIRTcrop[i])
##            print(VIRTcrop[i])
            o.append(Thread(target=Vrtsp_thread, args=(inframe, Nvc, VCoffset, VIRTcrop[i], VirtCamOffset+i, VIRTrtspURL[i])))  # for virtual camera
            o[i+Nonvif+Nrtsp].start()
            VCoffset+=Nvc
        FEoffset=FishEyeOffset
        for i in range(Nfisheye):
            Nfe=len(PTZparam[i])-1  # first entry is camera resolution, not PTZ view parameters
##            print(PTZparam[i])
            o.append(Thread(target=FErtsp_thread, args=(inframe, Nfe, FEoffset, PTZparam[i], FishEyeOffset+i, FErtspURL[i])))  # for virtual camera
            o[i+Nonvif+Nrtsp+Nvirt].start()
            FEoffset+=Nfe
        while threadsRunning < Nrtsp+Nvirt+Nfisheye:
            time.sleep(0.5)
        print("[INFO] All " + str(Nrtsp+Nvirt+Nfisheye) + " RTSP Camera Sampling Threads are running.")



    # *** start Coral TPU thread
    print("[INFO] starting Coral TPU AI Thread ...", )
    Ct=Thread(target=TPU_thread, args=(results, inframe, model, labels, Ncameras, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf))
    Ct.start()




    #*************************************************************************************************************************************
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    SEND_ALIVE=100  # send MQTT message approx. every SEND_ALIVE/fps seconds to reset external "watchdog" timer for auto reboot.
    waitCnt=0
    prevUImode=UImode
    currentDT = datetime.datetime.now()
    #start the FPS counter
    print("[INFO] starting the FPS counter ...")
    fps = FPS().start()
    print("[INFO] AI/Status: Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.publish("AI/Status", "Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    while not QUIT:
        try:
            try:
                (img, cami, personDetected, dt, ai, bp) = results.get(True,0.100)
            except:
                aliveCount = (aliveCount+1) % SEND_ALIVE   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
                if aliveCount == 0:
                    client.publish("AmAlive", "true", 0, False)
                waitCnt+=1
                img=None
                continue
            if img is not None:
                fps.update()    # update the FPS counter
                #personDetected=True   # force every frame to be written for testing, use with -d 0 or -d 1 option
                # setup for display or sending detection
                folder=dt.strftime("%Y-%m-%d")
                filename=dt.strftime("%H_%M_%S.%f")
##                filename=filename[:-5] + "_" + ai #just keep tenths, append AI source
                filename=filename[:-5] #just keep tenths, don't append AI source, fisheyeTPU is always TPU
                if localSave:
                    if __WIN__ is False:
                        lfolder=str(detectPath + "/" + folder)
                    else:
                        lfolder=str(detectPath + "\\" + folder)
                    if os.path.exists(lfolder) == False:
                        os.mkdir(lfolder)
                    if __WIN__ is False:
                        if personDetected:
                            outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) + "_" + AlarmMode  +"_AI.jpg")
                        else:   # in case saveAll option
                            outName=str(lfolder + "/" + filename + "_" + "Cam" + str(cami) + "_" + AlarmMode  +".jpg")
                    else:
                        if personDetected:
                            outName=str(lfolder + "\\" + filename + "_" + "Cam" + str(cami) + "_" + AlarmMode  +"_AI.jpg")
                        else:   # in case saveAll option
                            outName=str(lfolder + "\\" + filename + "_" + "Cam" + str(cami) + "_" + AlarmMode  +".jpg")

                    if (personDetected and not AlarmMode.count("Idle")) or saveAll:  # save detected image
                        cv2.imwrite(outName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if personDetected:
                    #outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                    outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + CamName[cami] +"_AI.jpg")
                    outName=outName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])    # for sending image as mqtt buffer, 10X+ less data being sent.
                    if retv:
                        client.publish(str(outName), bytearray(img_as_jpg), 0, False)
##                        print(outName)  # log detections
                    else:
                        print("[INFO] conversion of np array to jpg in buffer failed!")
                        continue
                # send image for live display in dashboard, convoluted, but trying ot minimize imencode() operations
                if ((CameraToView == cami) and (UImode == 1 or (UImode == 2 and personDetected))) or (UImode ==3 and personDetected):
                    if personDetected:
                        #topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +"_AI.jpg")
                        topic=str("ImageBuffer/!" + filename + "_" + CamName[cami] +"_AI.jpg")
                        client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                    else:
                        retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    # for sending image as mqtt buffer, 10X+ less data being sent.
                        if retv:
                            #topic=str("ImageBuffer/!" + filename + "_" + "Cam" + str(cami) +".jpg")
                            topic=str("ImageBuffer/!" + filename + "_" + CamName[cami] +".jpg")
                        else:
                            print("[INFO] conversion of np array to jpg in buffer failed!")
                            continue
                        client.publish(str(topic), bytearray(img_as_jpg), 0, False)
                # display the frame to the screen if enabled, in normal usage display is 0 (off)
                if dispMode > 0:
                    #name=str("Live_" + str(cami))
                    name=CamName[cami]
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
    print("[INFO] Run elapsed time: {:.2f} seconds.".format(fps.elapsed()))
    print("[INFO] Frames processed by AI system: " + str(fps._numFrames))
    print("[INFO] Main loop waited for results: " + str(waitCnt) + " times.")
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", "Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)


    # stop cameras
    if Nmqtt > 0:
        mqttCam.disconnect()
        mqttCam.loop_stop()
        for i in range(Nmqtt):
            print("MQTTcam/" + str(camList[i]) + " has dropped: " + str(mqttFrameDrops[i]) + " frames out of: " + str(mqttFrames[i]))

    # wait for threads to exit
    if Nonvif > 0:
        for i in range(Nonvif):
            o[i].join()
        print("[INFO] All Onvif Camera have exited ...")
    if Nrtsp > 0:
        for i in range(Nrtsp):
            o[i+Nonvif].join()
        print("[INFO] All rtsp Camera have exited ...")
    if Nvirt > 0:
        for i in range(Nvirt):
            o[i+Nonvif+Nrtsp].join()
        print("[INFO] All rtsp Camera have exited ...")
    if Nfisheye > 0:
        for i in range(Nfisheye):
            o[i+Nonvif+Nrtsp+Nvirt].join()
        print("[INFO] All rtsp Camera have exited ...")
    # stop TPU
    Ct.join()
    print("[INFO] All Coral TPU AI Thread has exited ...")


    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()


    # Send a blank image the dashboard UI
    print("[INFO] Clearing dashboard ...")
    img = np.zeros((imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (32,32,32)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    client.publish("ImageBuffer/!AI has Exited", bytearray(img_as_jpg), 0, False)
    time.sleep(1.0)


    # clean up localhost MQTT
    client.disconnect()     # normal exit, Will message should not be sent.
    currentDT = datetime.datetime.now()
    print("[INFO] Stopping MQTT Threads." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.loop_stop()      ### Stop MQTT thread


    # bye-bye
    currentDT = datetime.datetime.now()
    print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    print("")
    print("")



#####################################################################################################################################
#################################################### Thread Functions ###############################################################
#####################################################################################################################################


## *** Coral TPU Thread ***
#******************************************************************************************************************
#******************************************************************************************************************
# All spacial and "blob" false detection filtering is moved to the -mqtt controller host instead of being done here.
def TPU_thread(results, inframe, model, labels, Ncameras, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf):
    global QUIT
    global blobThreshold    # so far, MobileNet-SSDv2 hasn't needed the blob filter, needed 20FEB2020wbk
    waits=0
    drops=0
    fcnt=0
    cq=0
    nextCamera=0
    ai = "TPU"
    cfps = FPS().start()
##    # the region filter can also be done in the node-red instead, doing it here is easier for only one rtsp stream and two virtual cameras.
##    poly = [
##        [[50,0],[0,1280],[1280,430],[0,350]],
##        [[0,0],[1280,0],[480,440],[0,440]],
##        [[0,120],[960,150],[960,580],[0,480]],
##        [[0,120],[960,150],[960,580],[0,480]]
##    ]
    while not QUIT:
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
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
        if edgetpu_version < '2.11.2':
            detection = model.DetectWithImage(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
        else:
            detection = model.detect_with_image(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
        cfps.update()    # update the FPS counter
        fcnt+=1
        ##imageDT = datetime.datetime.now()
        # loop over the detection results
        boxPoints=(0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, Xcenter, Ycenter, Xlength, Ylength
        for r in detection:
            if r.label_id == 0:
                # extract the bounding box and box and predicted class label
                box = r.bounding_box.flatten().astype("int")
                label = labels[r.label_id]
                initialConf=r.score
                (startX, startY, endX, endY) = box.flatten().astype("int")
                X_MULTIPLIER = float(w) / PREPROCESS_DIMS[0]
                Y_MULTIPLIER = float(h) / PREPROCESS_DIMS[1]
                startX = int(startX * X_MULTIPLIER)
                startY = int(startY * Y_MULTIPLIER)
                endX = int(endX * X_MULTIPLIER)
                endY = int(endY * Y_MULTIPLIER)
##                # Screen for lower right of bounding box inside region of interest, can do here or in node-red
##                if point_inside_polygon(endX,endY,poly[cam]):
##                    boxPoints=(startX,startY, endX,endY)
##                else:
##                    continue
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                xlen=endX-startX
                ylen=endY-startY
                if float(xlen*ylen)/(w*h) > blobThreshold:     # detection filling too much of the frame is bogus
                   continue
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                # draw the bounding box and label on the image
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} TPU".format(initialConf * 100,
                    str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY))
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
            if edgetpu_version < '2.11.2':
                detection = model.DetectWithImage(frame, threshold=verifyConf, keep_aspect_ratio=True, relative_coord=False)
            else:
                detection = model.detect_with_image(frame, threshold=verifyConf, keep_aspect_ratio=True, relative_coord=False)
            cfps.update()    # update the FPS counter
            # loop over the detection results
            boxPointsV = (0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, 0, 0, 0, 0 only first four are used for dbg plots
            for r in detection:
              if r.label_id == 0:
                  if r.score > verifyConf:
                        personDetected = True
                        text = "Verify: {:.1f}%".format(r.score * 100)   # show verification confidence
                        cv2.putText(image, text, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        break    # only need one verification
        else:
            ndetected=0     # flag that verification not needed
        # Queue results
        try:
            if personDetected:
                results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 1.0)    # try not to drop frames with detections
            else:
                # change image to zoom if you don't want unverified detection box on displayed/saved image, at low frame rates the box is informative
##                results.put((image, cam, personDetected, imageDT, "Coral"), True, 0.033)
                results.put((image, cam, personDetected, imageDT, ai, boxPoints), True, 0.016)
        except:
            # presumably results queue was full, main thread too slow.
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("Coral TPU thread waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))




## Fisheye Window snippet
# --> https://github.com/daisukelab/fisheye_window
class FishEyeWindow(object):
    """Fisheye Window class
    You can get image out of your fisheye image for desired view.
    1. Create instance by feeding image sizes.
    2. Call buildMap to set the view you want.
       This calculates the map for the 'remap.'
    3. Call getImage that simply remaps.
    """
    def __init__(
            self,
            srcWidth,
            srcHeight,
            destWidth,
            destHeight
        ):
        # Initial parameters
        self._srcW = srcWidth
        self._srcH = srcHeight
        self._destW = destWidth
        self._destH = destHeight
        self._al = 0
        self._be = 0
        self._th = 0
        self._R  = srcWidth / 2
        self._zoom = 1.0
        # Map storage
        self._mapX = np.zeros((self._destH, self._destW), np.float32)
        self._mapY = np.zeros((self._destH, self._destW), np.float32)
    def buildMap(self, alpha=None, beta=None, theta=None, R=None, zoom=None):
        # Set the angle parameters
        self._al = (alpha, self._al)[alpha == None]
        self._be = (beta, self._be)[beta == None]
        self._th = (theta, self._th)[theta == None]
        self._R = (R, self._R)[R == None]
        self._zoom = (zoom, self._zoom)[zoom == None]
        # Build the fisheye mapping
        al = self._al / 180.0
        be = self._be / 180.0
        th = self._th / 180.0
        A = np.cos(th) * np.cos(al) - np.sin(th) * np.sin(al) * np.cos(be)
        B = np.sin(th) * np.cos(al) + np.cos(th) * np.sin(al) * np.cos(be)
        C = np.cos(th) * np.sin(al) + np.sin(th) * np.cos(al) * np.cos(be)
        D = np.sin(th) * np.sin(al) - np.cos(th) * np.cos(al) * np.cos(be)
        mR = self._zoom * self._R
        mR2 = mR * mR
        mRsinBesinAl = mR * np.sin(be) * np.sin(al)
        mRsinBecosAl = mR * np.sin(be) * np.cos(al)
        centerV = int(self._destH / 2.0)
        centerU = int(self._destW / 2.0)
        centerY = int(self._srcH / 2.0)
        centerX = int(self._srcW / 2.0)
        # Fill in the map, slows dramatically with large view (destination) windows
        for absV in range(0, int(self._destH)):
            v = absV - centerV
            vv = v * v
            for absU in range(0, int(self._destW)):
                u = absU - centerU
                uu = u * u
                upperX = self._R * (u * A - v * B + mRsinBesinAl)
                lowerX = np.sqrt(uu + vv + mR2)
                upperY = self._R * (u * C - v * D - mRsinBecosAl)
                lowerY = lowerX
                x = upperX / lowerX + centerX
                y = upperY / lowerY + centerY
                _v = (v + centerV, v)[centerV <= v]
                _u = (u + centerU, u)[centerU <= u]
                self._mapX.itemset((_v, _u), x)
                self._mapY.itemset((_v, _u), y)

    def getImage(self, img):
        # Look through the window
        output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_LINEAR)
        #output = cv2.remap(img, self._mapX, self._mapY, cv2.INTER_CUBIC) # no significant improvement on 4 Mpixel test image
        return output


# create virtual cameras from PTZ crops from a fisheye camera rtsp stream
# Note the PTZ param are string variables read from the fisheye.rtsp text file
# created with the interactive fisheye_window C++ utility program.
def FErtsp_thread(inframe, Nfe, FEoffset, PTZparam, camn, URL):
    global QUIT
    global threadLock
    global threadsRunning
    ocnt=[]
    for i in range(Nfe):
        ocnt.append(0)      # init counter array
    fe=[]
    Error=False
    Error2=False
##    print(PTZparam)
    threadLock.acquire()
    mapFilename="fisheye" +str(camn)+ "_map"
    try:
      filehandler = open(mapFilename, 'rb')
      currentDT = datetime.datetime.now()
      print( "Loading " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      fe = pickle.load(filehandler)
      filehandler.close()
    except:
      currentDT = datetime.datetime.now()
      print( "Creating " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      for i in range(Nfe):
        if i == 0:
            srcW=int(PTZparam[0][0])
            srcH=int(PTZparam[0][1])
        # PTZparam = [ [srcW,srcH], [destW, destH, alpha, beta, theta, zoom], [...] ] chosen with fisheye_window  C++ utility
        print("FE" +str(camn)+ " PTZview" +str(i)+ " " +str(PTZparam[i+1]))
        fe.append(FishEyeWindow(srcW, srcH, int(PTZparam[i+1][0]), int(PTZparam[i+1][1])))    # instance a view with desired output image size
        fe[i].buildMap(alpha=float(PTZparam[i+1][2]), beta=float(PTZparam[i+1][3]),
                       theta=float(PTZparam[i+1][4]), zoom=float(PTZparam[i+1][5]))    # build map for this PTZ view
      currentDT = datetime.datetime.now()
      print("Saving FE" +str(camn)+ " virtual PTZ views as: " + mapFilename + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
      filehandler = open(mapFilename, 'wb')
      pickle.dump(fe, filehandler)
      filehandler.close()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye Camera RTSP stream FE" + str(camn) + " is opening..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
#    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] Fisheye RTSP stream sampling thread" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
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
                    print('[Error!] RTSP Camera FE'+ str(camn) + ': ' + URL[0:33] + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
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
                        print('[Error2!] RTSP stream FE'+ str(camn) + ' re-open failed! $$$ ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                        print('*** Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(5.0)
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera FE'+ str(camn) + ' has recovered: ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream FE'+ str(camn) + ': ' + str(e) + " " + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)

        if frame is not None:
            imageDT = datetime.datetime.now()
            for i in range(Nfe):
                try:
                    if inframe[FEoffset+i].full():
                        [_,_,_]=inframe[FEoffset+i].get(False)    # remove oldest sample to make space in queue
                        ocnt[i]+=1   # it this happens here, it shouldn't happen below
                    PTZview=fe[i].getImage(frame)
                    inframe[FEoffset+i].put((PTZview, FEoffset+i, imageDT), True)  ## force this frame to complete in all queues
                except: # most likely queue is full, Python queue.full() is not 100% reliable
                    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
                    ocnt[i]+=1

    Rcap.release()
    print("RTSP Fisheye Camera sampling thread" + str(camn) + " is exiting ...")
    for i in range(Nfe):
        print("   Fisheye Cam "+ str(FEoffset+i) +" dropped frames " + str(ocnt[i]) + " times.")






# *** RTSP Sampling Thread modified for virtual camera cropping
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def Vrtsp_thread(inframe, Nvc, VCoffset, Crops, camn, URL): # for making one camera into muiltiple virtual cameras
#def rtsp_thread(inframe, camn, URL):
    global QUIT
    global threadLock
    global threadsRunning
    ocnt=[]
    Error=False
    Error2=False
    for i in range(Nvc):
        ocnt.append(0)
    currentDT = datetime.datetime.now()
    print("[INFO] Virtual Camera RTSP stream sampling thread" + str(camn) + " is starting..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] Virtual Camera RTSP stream sampling thread" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
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
                    print('[Error!] Virtual Camera RTSP Camera'+ str(camn) + ': ' + URL[0:33] + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
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

        if frame is not None:
            imageDT = datetime.datetime.now()
            for i in range(Nvc):
                try:
                    if inframe[VCoffset+i].full():
                        [_,_,_]=inframe[VCoffset+i].get(False)    # remove oldest sample to make space in queue
                        ocnt[i]+=1   # it this happens here, it shouldn't happen below
###                    inframe[VCoffset+i].put((frame[ Crops[i][1]:Crops[i][3], Crops[i][0]:Crops[i][2] ], VCoffset+i, imageDT), False)   # no block if queue full, go grab fresher frame
                    img=frame[ Crops[i][1]:Crops[i][3], Crops[i][0]:Crops[i][2] ]
                    clone=img.copy()
                    inframe[VCoffset+i].put((clone, VCoffset+i, imageDT), True)  ## force this frame to complete in all queues
                except: # most likely queue is full, Python queue.full() is not 100% reliable
                    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
                    ocnt[i]+=1

    Rcap.release()
    print("RTSP Virtual Camera sampling thread" + str(camn) + " is exiting ...")
    for i in range(Nvc):
        print("   Virt Cam "+ str(VCoffset+i) +" dropped frames " + str(ocnt[i]) + " times.")


# determine if a point is inside a given polygon or not
# code from:    http://www.ariel.com.au/a/python-point-int-poly.html
# algorthim:    http://paulbourke.net/geometry/polygonmesh/
#
# Polygon is a list of (x,y) pairs.
def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside



# *** RTSP Sampling Thread
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
### So far none of the IOT class ARM computers can handle more tha 4 rtsp threads
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
                imageDT = datetime.datetime.now()
                if inframe.full():
                    [_,_,_]=inframe.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below
                inframe.put((frame, camn, imageDT), False)   # no block if queue full, go grab fresher frame
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
            time.sleep(5.0)     # let other threads have more time while this camera recovers, which sometimes takes minutes
        try:
            if frame is not None:
                imageDT = datetime.datetime.now()
                inframe.put((frame, camn, imageDT), True, 0.200)
        except: # most likely queue is full
            ocnt=ocnt+1
            ##time.sleep(sleepyTime)
            continue
    print("ONVIF Camera" + str(camn) + " thread is exiting, dropped frames " + str(ocnt) + " times.")



# python boilerplate
if __name__ == '__main__':
    main()


