#!/usr/bin/env python3
#
### Pi4TPU.py 16AUG2019wbk
## Requires Pi4TPU_AI_Controller.json flow running in node-red to save images.  Local save option has been removed.
## Use AI_dev.py or TPU.py if you want local save option.
#
# Derived from AI_dev.py
# This is basically AI_dev with everything removed except for the TPU AI thread, Onvif, rtsp, and mqttCam threads.  All code
# is moved back into this single python file.  Not sure it ends up any different from TPU.py except for some Coral development
# board "Mendel" ID strings, and the local saving of detection images being removed.
#
## First steps to making AI person detection "appliance".  Run the AI and send detections to central MQTT broker.
#
## 17AUG2019wbk
# Add detection box points (startX, startY, endX, endY) as part of MQTT Topic string for possible post processing detection filter.
# Reorginize main loop to avoid imwrite() and imencode() if results are not going to be used.
#
# 17OCT2019wbk -- Add syncronized wait to rtsp thread startup.
##
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
## 27DEC2019wbk, add PiCamera Module support, change some command argument defaults and names:
# tested PiCamera Module support on Pi3B with NCS and OpenVINO:
#  ./Pi4TPU.py -mqtt kahuna -pi  --> get ~8.3 fps


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
from threading import Thread, Lock

# TPU
from edgetpu.detection.engine import DetectionEngine


if True:
    # *** get command line parameters
    # construct the argument parser and parse the arguments for this module
    ap = argparse.ArgumentParser()
        
    ap.add_argument("-c", "--confidence", type=float, default=0.70, help="detection confidence threshold")
    ap.add_argument("-vc", "--verifyConfidence", type=float, default=0.80, help="detection confidence for verification")
    ap.add_argument("-nvc", "--noVerifyConfidence", type=float, default=.98, help="initial detection confidence to skip verification")
    ap.add_argument("-blob", "--blobFilter", type=float, default=.20, help="reject detections that are more than this fraction of the frame")
    ap.add_argument("-dbg", "--debug", action="store_true", help="display images to debug detection verification thresholds")
    
    # specify text file with list of URLs for camera rtsp streams
    ap.add_argument("-rtsp", "--rtspURLs", default="MYcameraURL.rtsp", help="path to file containing rtsp camera stream URLs")
    
    # specify text file with list of URLs cameras http "Onvif" snapshot jpg images
    ap.add_argument("-cam", "--cameraURLs", default="MYcameraURL.txt", help="path to file containing http camera jpeg image URLs")
    
    # display mode, mostly for test/debug and setup, general plan would be to run "headless"
    ap.add_argument("-d", "--display", type=int, default=0,
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
    
    # specify MQTT broker
    ap.add_argument("-sys", "--systemID", default="Pi4TPU", help="name of this system used in detection filenames.")
   
    # PiCamera module
    ap.add_argument("-pi", "--PiCam", action="store_true", help="Use Pi  camera module")
   
    args = vars(ap.parse_args())
    PiCAM = args["PiCam"]


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
global sysIDstr
global dbg
global CamName


## MendelTPU is specific for my use
# Map Lorex camera names to camera numbers, Lorex uses 1-16, Python uses 0-15
## After Lorex died, I reorderd the cameras on the Qcamera DVR-16 replacement for "better" mosaic display
LorexName = [
    "MailBox",
    "HummingbirdLeft",
    "FrontDoor",
    "HummingbirdRight",
    "CliffwoodRight",
    "CliffwoodLeft",
    "DriveWay",
    "Shed",
    "KitchenDoor",
    "Garage",
    "Patio",
    "SideYard",
    "PoolEquipment",
    "PoolDeck",
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
    "Cam7"
]


# *** constants for MobileNet-SSD & MobileNet-SSD_V2  AI models
# frame dimensions should be sqaure for MobileNet-SSD
PREPROCESS_DIMS = (300, 300)


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
        print(str(msg.topic)+":  " + str(msg.payload) + currentDT.strftime(" ... %Y-%m-%d %H:%M:%S"))
        AlarmMode = str(msg.payload)
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
    global sysIDstr
    global dbg
    global CamName
    global blobThreshold    
       

    # set variables from command line auguments or defaults
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
    sysIDstr = args["systemID"]   
    PiCAM = args["PiCam"]
    

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


    # *** allocate queues
    # we simply make one queue for each camera, rtsp stream, and MQTTcamera
    QDEPTH = 2      # small values improve latency
##    QDEPTH = 1      # small values improve latency
    print("[INFO] allocating camera and stream image queues...")
    if PiCAM:
        PiCamOffset=Ncameras
        Ncameras+=1
        print("[INFO] allocating queue for PiCamera Module...")
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
                #name=str("Live_" + str(i))
                name=CamName[i]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nrtsp > 0:
            print("[INFO] setting up rtsp camera image windows ...")
            for i in range(Nrtsp):
                #name=str("Live_" + str(i+Nonvif))
                name=CamName[i+Nonvif]
                cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        if Nmqtt > 0:
            print("[INFO] setting up MQTT camera image windows ...")
            for i in range(Nmqtt):
                #name=str("Live_" + str(i+mqttCamOffset))
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
    client.will_set("AI/Status", sysIDstr + " Python AI has died!", 2, True)  # let everyone know we have died, perhaps node-red can restart it
    client.connect(MQTTserver, 1883, 60)
    client.loop_start()

    # *** MQTT send a blank image to the dashboard UI
    # build grey image for mqtt windows
    img = np.zeros(( imwinHeight, imwinWidth, 3), np.uint8)
    img[:,:] = (127,127,127)
    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    print("[INFO] Clearing dashboard ...")
    client.publish("ImageBuffer/!"+ sysIDstr + " AI has Started.", bytearray(img_as_jpg), 0, False)


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
        time.sleep(0.1)     # force thread dispatch
        for i in camList:
            mqttCam.publish(str("sendOne/" + str(i)), "", 0, False)   # start messages flowing
            
            
            
    # *** start camera reading threads
    o = list()
    if Nonvif > 0:
        print("[INFO] starting " + str(Nonvif) + " Onvif Camera Threads ...")
        for i in range(Nonvif):
            o.append(Thread(target=onvif_thread, args=(inframe[i], i, CameraURL[i])))
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
        print("[INFO] starting " + str(Nrtsp) + " RTSP Camera Sampling Threads ...")
        for i in range(Nrtsp):
            o.append(Thread(target=rtsp_thread, args=(inframe[i+Nonvif], i, rtspURL[i])))
            o[i+Nonvif].start()
        while threadsRunning < Nrtsp:
            time.sleep(0.5)
        print("[INFO] All " + str(Nrtsp) + " RTSP Camera Sampling Threads are running.")



    # *** start Coral TPU thread
    print("[INFO] starting "+sysIDstr+" Coral TPU AI Thread ...", )
    Ct=Thread(target=TPU_thread, args=(results, inframe, model, labels, Ncameras, PREPROCESS_DIMS, confidence, noVerifyNeeded, verifyConf))
    Ct.start()




    #*************************************************************************************************************************************
    # *** enter main program loop (main thread)
    # loop over frames from the camera and display results from AI_thread
    excount=0
    aliveCount=0
    waitCnt=0
    prevUImode=UImode
    currentDT = datetime.datetime.now()
    print("[INFO] AI/Status: Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
    client.publish("AI/Status", sysIDstr + " Python AI running." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)
    #start the FPS counter
    print("[INFO] starting the FPS counter ...")
    fps = FPS().start()
    while not QUIT:
        try:
            try:
                (img, cami, personDetected, dt, ai, bp) = results.get(True,0.100)
            except:
                aliveCount = (aliveCount+1) % 200   # MQTTcam images stop while Lorex reboots, recovers eventually so keep alive
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
                filename=filename[:-5] + "_" + ai #just keep tenths, append AI source
                if personDetected:
                    #outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + "Cam" + str(cami) +".jpg")
                    outName=str("AIdetection/!detect/" + folder + "/" + filename + "_" + CamName[cami] +".jpg")
                    outName=outName + "!" + str(bp[0]) + "!" + str(bp[1]) + "!" + str(bp[2]) + "!" + str(bp[3]) + "!" + str(bp[4]) + "!" + str(bp[5]) + "!" + str(bp[6]) + "!" + str(bp[7])
                    retv, img_as_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])    # for sending image as mqtt buffer, 10X+ less data being sent.
                    if retv:
                        client.publish(str(outName), bytearray(img_as_jpg), 0, False)
                        print(outName)  # log detections
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
                aliveCount = (aliveCount+1) % 200
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
    print("[INFO] Main looped waited for results: " + str(waitCnt) + " times.")
    currentDT = datetime.datetime.now()
    client.publish("AI/Status", sysIDstr + " Python AI stopped." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"), 2, True)

   
    # stop cameras 
    if Nmqtt > 0:
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
    client.publish("ImageBuffer/!"+ sysIDstr + " AI has Exited", bytearray(img_as_jpg), 0, False)
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
    global sysIDstr
    global dbg
    global blobThreshold    # so far, MobileNet-SSDv2 hasn't needed the blob filter.
    waits=0
    drops=0
    fcnt=0
    cq=0
    nextCamera=0
    cfps = FPS().start()
    while not QUIT: 
        cq=nextCamera
        nextCamera = (nextCamera+1)%Ncameras
        # get a frame
        try:
            (image, cam) = inframe[cq].get(True,0.100)
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
        detection = model.DetectWithImage(frame, threshold=confidence, keep_aspect_ratio=True, relative_coord=False)
        cfps.update()    # update the FPS counter
        fcnt+=1
        imageDT = datetime.datetime.now()
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
                boxPoints=(startX,startY, endX,endY)
                xlen=endX-startX
                ylen=endY-startY
                if float(xlen*ylen)/(w*h) > blobThreshold:     # detection filling too much of the frame is bogus
                   continue
                xcen=int((startX+endX)/2)
                ycen=int((startY+endY)/2)
                boxPoints=(startX,startY, endX,endY, xcen,ycen, xlen,ylen)
                # draw the bounding box and label on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                label = "{:.1f}%  C:{},{}  W:{} H:{}  UL:{},{}  LR:{},{} {} TPU".format(initialConf * 100, 
                        str(xcen), str(ycen), str(xlen), str(ylen), str(startX), str(startY), str(endX), str(endY), sysIDstr)
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
            detection = model.DetectWithImage(frame, threshold=verifyConf, keep_aspect_ratio=True, relative_coord=False)
            cfps.update()    # update the FPS counter
            # loop over the detection results
            imgDT = datetime.datetime.now()
            tlabel = "{:.1f}%  Coral".format(initialConf * 100)
            cv2.putText(img, tlabel, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            boxPointsV = (0,0, 0,0, 0,0, 0,0)  # startX, startY, endX, endY, 0, 0, 0, 0 only first four are used for dbg plots
            for r in detection:
              if r.label_id == 0:
                  if dbg:
                        box = r.bounding_box.flatten().astype("int")
                        (startX, startY, endX, endY) = box.flatten().astype("int")
                        boxPointsV = (startX,startY, endX,endY, 0,0, 0,0)
                        label = labels[r.label_id]
                        text = "{}: {:.1f}%".format(label, r.score * 100)
                        cv2.putText(img, text, (2, min(270,endY)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
                        # draw the person bounding box and label on the image
                        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
                        text = "{:.1f}% CoralV".format(r.score * 100)
                        cv2.putText(img, text, (2, (h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
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
                results.put((image, cam, personDetected, imageDT, sysIDstr, boxPoints), True, 1.0)    # try not to drop frames with detections
            else:
                if dbg is True and ndetected == 1:  # I want to see what the "zoom" has rejected
                    results.put((img, cam, True, imgDT, "CoralV", boxPointsV), True, 1.0) # force zoom rejection file write
                # change image to zoom if you don't want unverified detection box on displayed/saved image, at low frame rates the box is informative
##                results.put((image, cam, personDetected, imageDT, "Coral"), True, 0.033)
                results.put((image, cam, personDetected, imageDT, sysIDstr, boxPoints), True, 0.016)
        except:
            # presumably results queue was full, main thread too slow.
            drops+=1
            continue
    # Thread exits
    cfps.stop()    # stop the FPS counter timer
    print("Coral TPU thread waited: " + str(waits) + " dropped: " + str(drops) + " out of "
         + str(fcnt) + " images.  AI: {:.2f} inferences/sec".format(cfps.fps()))





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
                inframe.put((frame, camn), True, 0.200)
                ##time.sleep(sleepyTime)   # force thread switch, hopefully smoother sampling, 10Hz seems upper limit for snapshots
        except: # most likely queue is full
            ocnt=ocnt+1
            ##time.sleep(sleepyTime)
            continue
    print("ONVIF Camera" + str(camn) + " thread is exiting, dropped frames " + str(ocnt) + " times.")




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
                if inframe.full():
                    [_,_]=inframe.get(False)    # remove oldest sample to make space in queue
                    ocnt+=1     # if happens here shouldn't happen below     
                inframe.put((frame, camn), False)   # no block if queue full, go grab fresher frame
        except: # most likely queue is full
            ocnt+=1          
    # a large drop count for rtsp streams is not a bad thing as we are trying to keep the input buffers nearly empty to reduce latency.
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
# its ran for over 10 weeks with only a single error, from which it automatically recovered.

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
                try:
                  if self.inQueue.full():
                    [_,_]=self.inQueue.get(False)    # remove oldest sample to make space in queue
                    self.ocnt+=1     # if happens here shouldn't happen below     
                  self.inQueue.put((self.frame, self.camn), False)   # no block if queue full, go grab fresher frame
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


