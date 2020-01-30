#!/usr/bin/env python3
#
# rtsp2mqtt.py
#
## 13Dec2019wbk Appears to be an issue with UHD cameras when running this on a remote host.  Something about mqtt network transport?
## All fine when run on same (fast) machine, but the UHD camera latency quickly increases to ~10-15 seconds when split to two machines!
## All on localhost i7-6700K:  get ~38.9 fps with AI_dev.py -Nmqtt 15 -mqttDemand,  without -mqttDemand get ~44.1 fps, but UHD latency grows.
## Very poor performance on when all on slower machines like i7-4500U, so its not 100% network issue.
## With AI_dev.py on i7-4500U -Nmqtt 15 -mqttDemand  and rtsp2mqtt on i7-6700K get ~31.5 fps, without -mqttDemand get ~25.8 fps and UHD camera 
## latency increases to ~10+ seconds,  performance issues are not that simple!  Giving up for now.  Perhaps the issue is Python "pickling"
## of the binary buffers to send and receive via MQTT
##
## for reference, with -rtsp option in AI_dev instead of -Nmqtt get ~44.9 fps on i7-6700K.
##
## I expected this to help with weaker machines running the AI to seperate the RTSP decoding into seperate processes or on a second
## machine, but seems not.
##
#
## 11DEC2019wbk Make onDemand option to wait for sendOne messages before sending MQTTcam/N mqtt message.
## Normally sends mqtt QOS0 message for each frame decoded from the rtsp stream.  This can overwhelm IOT class CPUs (Pi4B etc.)
##  using --onDemand (-od) is an attempt to sort of emulate Onvif snapshots.
#
#
## 11SEPT2019wbk
# Lorex DVR Died.  Modified URLs for Qcamera DVR-16 replacement.
# Its H.265+ so regularly throws "Invalid UE golomb code" error/warning. launch with 2>/dev/null 
# Recovered after Qcamera DVR reboot which took less than 3 minutes total
#
#
# 17JUL2019wbk
# OpenCV Python code to grab rtsp stream frames and distribute as MQTT buffers.
# Derived from rtsp2mqtt.py modified to use multiprocessing instead of threading
#
#
# 18JUL2019wbk  Launch mqtt thread from rtsp thread and eliminate use of mp.queue by removing debugging display option
# 19JUL2019wbk  Recovered after unplugging and replugging a POE netcam!
# Messages:
###   [Error!] RTSP Camera3: rtsp://admin:admin@192.168.2.67/m 2019-07-19 10:21:13
###   *** Will close and re-open Camera3 RTSP stream in attempt to recover.
###   [Error2!] RTSP stream3 re-open failed! $$$ rtsp://admin:admin@192.168.2.67/m ---  2019-07-19 10:21:27
###   *** Will loop closing and re-opening Camera3 RTSP stream, further messages suppressed.
###   [$$$$$$] RTSP Camera3 has recovered: rtsp://admin:admin@192.168.2.67/m ---  2019-07-19 10:21:59
#
# I belive this version is a winner!  Running on i7-4500U miniPC it served up ~59.8 fps to i7-6700K desktop
# running: AI_dev.py -camMQTT i7AI -Nmqtt 15 -nTPU 1  (Note, these were all 1080p 5 fps rtsp streams, adding 4K cameras caused issues!)
#
#
# only tested with python3 
# starting as:
#   python3 rtsp2mqttP.py 2>/dev/null
# is usefull to stop seeing the warnings from Chinese netcams, python3 rtsp2mqtt.py 2>logfile.txt will let you look at the jibber-jabber
#
#
## 22JUN2019wbk  rtsp2mqtt.py recovered after a Lorex spontaneous reboot!  
### AI_OVmt.py reading the same Lorex rtsp streams crashed with Inference Engine exception! 30 rtsp connections to the Lorex
### might have had somehting to do with it.  Need to repeat the test with only AI_OVmt.py running and reboot the Lorex DVR.
### Also need to double check that the rtsp error handling is the same in both programs.
#
### 9AUG2019wbk  spin-off demand version
# This worked great with clients on fast hardware, but end up buffering the QoS 0 messages in the network stack on Pi4 etc.
# Modify to drop rtsp frames until a request is received, then push out the next frame, to work sort of like Onvif snapshot
# 12AUG2019 change everything to QOS 0
#
### on i7-6700K -nTPU 1 -camList 1 2 3 5 5 14 -camMQTT i5ai gets: ~24.4 fps  ~30 fps would be every frame from every camera
##              -nTPU -rtsp 6Lorex.txt                      gets: ~29.5 fps.
#
#
### 14AUG2019wbk  This code recovered and resumed through a Lorex DVR firmware upgrade and reboot cycle!


import sys, traceback
import cv2
from imutils.video import FPS
import os
import signal
import datetime
import time
import paho.mqtt.client as mqtt
# threading stuff
import argparse
import multiprocessing as mp


# for wait until all Processes are running
global threadsRunning
global threadLock
threadLock = mp.Lock()
threadsRunning = mp.Value('i',0)


# mark start of this code in log file
print("**************************************************************")
currentDT = datetime.datetime.now()
print("*** " + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
print("[INFO] using openCV-" + cv2.__version__)


ap = argparse.ArgumentParser()
# specify text file with list of URLs for camera rtsp streams
ap.add_argument("-rtsp", "--rtspURLs", default="MYcameraURL.rtsp", help="path to file containing rtsp camera stream URLs")
ap.add_argument("-od", "--onDemand", action="store_true", help="Wait for sendOne messages")
args = vars(ap.parse_args())

onDemand = args["onDemand"]
if onDemand:
    print("[INFO] startup option: --onDemand  means mqtt image buffers will only be sent after")
    print("       receiving sendOne/N messages, instead of sending immediately.")
else:
    print("[INFO] mqtt image buffers will be sent immediately, this can overwhelm IOT class CPUs.")
    

# *** get rtsp URLs
RTSP = args["rtspURLs"]
try:
    rtspURL=[line.rstrip() for line in open(RTSP)]
    Nrtsp=len(rtspURL)
    rtspError=list()
    print("[INFO] " + str(Nrtsp) + " rtsp stream processes will be created.")
except:
    # fallback to trying cameras in my test setup
    print("[INFO] No " + str(RTSP) + " file.  Falling back to using built-in debugging defaults.")
    print("This, of course is unlikely to work for your system!")
    ## fall through to use my default selection
    # Choose one!
    useLorex = False
##    useLorex = True
##  Catastrophic failure of my Lorex DVR found a Qcamera DVR-16 that had good compatability  
    if useLorex:
        # my Lorex LHV2016 DVR (TV-CVI analog cameras)  the cameras are set for 5 fps
        # all 15 cameras on my i7 Desktop (quad core hyperthreading) achieves ~75 fps
        rtspURL= [
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=1&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=2&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=3&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=4&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=5&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=6&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=7&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=8&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=9&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=10&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=11&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=12&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=13&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=14&subtype=0",
            "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=15&subtype=0"##,
        ##    "rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=16&subtype=0",               # not used at present
        ]
    else:   # URLS for my Qcameera DVR-16 replacement system, kept same cameras and cables, great deal for $110  
        rtspURL= [
            # alternate cameras for test/debug
            ##"rtsp://admin:355113@192.168.2.164:554/cam/realmonitor?channel=4&subtype=1",  # Lorex DVR substream, lame CIF
            ##"rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov",     # public animation stream
            ##"rtsp://b1.dnsdojo.com:1935/live/sys3.stream",   # public beach scene
            # no reason they can't be mixed.
            ##"rtsp://192.168.2.156:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream",
            ##"rtsp://192.168.2.157:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream",
            ##"rtsp://192.168.2.126:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream",
            ##"rtsp://admin:admin@192.168.2.67/media/video1",
            ##"rtsp://192.168.2.53:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream" # straight fom China Besder 8mm, not recommended!
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=1&stream=0.sdp",    # cheapo QcamDVR
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=2&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=3&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=4&stream=0.sdp", 
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=5&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=6&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=7&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=8&stream=0.sdp", 
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=9&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=10&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=11&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=12&stream=0.sdp", 
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=13&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=14&stream=0.sdp",
            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=15&stream=0.sdp"##,
##            "rtsp://192.168.2.97:554/user=admin&password=355/113&channel=16&stream=0.sdp" 
        ]
    Nrtsp=len(rtspURL)



global QUIT
QUIT=False  # True exits main loop and all threads
global sendOne  # True sends next rtsp frame as jpeg image in MQTT buffer
sendOne=[]
for i in range(Nrtsp):
    sendOne.append(mp.Value('i',False)) # I think this needs to be made process/thread safe


# Boilerplate code to setup signal handler for graceful shutdown on Linux
# python process scoping rules are sublte I'm trying to squeeze out maximum performance by minimizing IPC
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



# *** RTSP Sampling Process
#******************************************************************************************************************
# rtsp stream sampling thread
### 20JUN2019 wbk much improved error handling, can now unplug & replug a camera, and the thread recovers
def rtsp_process(camn, URL):
    global QUIT
    global sendOne
    global threadsRunning
    global threadLock
    ocnt=0
    Error=False
    Error2=False
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling process" + str(camn) + " is starting..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    #print("       connecting to localhost broker for MQTT camera input" + str(camn))
    mqttCam = mqtt.Client(userdata=camn)
    mqttCam.on_connect = on_connect
    mqttCam.on_message = on_message
    mqttCam.on_publish = on_publish
    mqttCam.on_disconnect = on_disconnect
    mqttCam.connect("localhost", 1883, 60)
    mqttCam.loop_start()
    Rcap=cv2.VideoCapture(URL)
    Rcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # doesn't throw error or warning in python3, but not sure it is actually honored
    threadLock.acquire()
    currentDT = datetime.datetime.now()
    print("[INFO] RTSP stream sampling process" + str(camn) + " is running..." + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    threadsRunning.value += 1
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
                    ocnt+=1
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
                        ocnt+=1                   
                        currentDT = datetime.datetime.now()
                        print('[Error2!] RTSP stream'+ str(camn) + ' re-open failed! $$$ ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                        print('*** Will loop closing and re-opening Camera' + str(camn) +' RTSP stream, further messages suppressed.')
                    time.sleep(30.0) ## takes about 4 minutes to recover when Lorex apparetly auto reboots.
                continue
            if gotFrame: # path for sucessful frame grab, following test is in case error recovery is in progress
                if Error:   # log when it recovers
                    currentDT = datetime.datetime.now()
                    print('[$$$$$$] RTSP Camera'+ str(camn) + ' has recovered: ' + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
                    Error=False    # after geting a good frame, enable logging of next error
                    Error2=False
                if frame is not None:
                    if onDemand and not sendOne[camn].value:
                        continue    # drop this frame
                    retv, img_as_jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])    # for sending image as mqtt buffer, 10X+ less data being sent.
                    if not retv:
                        print("[INFO] thread{} conversion of np array to jpg in buffer failed!", str(camn))
                        img_as_jpg = None
                    else:
                        if onDemand:
                            sendOne[camn].value=False
                        mqttCam.publish(str("MQTTcam/" + str(camn)), bytearray(img_as_jpg), 0, False)
##            time.sleep(0.01)  # force dispatch in attempt improve smoothness
        except KeyboardInterrupt:
            QUIT=True
            continue
        except Exception as e:
            frame = None
            currentDT = datetime.datetime.now()
            print('[Exception] RTSP stream'+ str(camn) + ': ' + str(e) + " " + URL[0:33] + ' --- ' + currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
            time.sleep(10.0)
    Rcap.release()
    mqttCam.loop_stop()
    print("RTSP stream sampling process" + str(camn) + " is exiting, errors: " + str(ocnt) + ".")




# *** connect to MQTT broker
def on_publish(client, camn, mid):
    ##global sendOne
    ##sendOne[camn].value=False
    #print("mid: " + str(mid))      # don't think I need to care about this for now, print for initial tests
    pass


def on_disconnect(client, camn, rc):
    global sendOne
    if rc != 0:
        currentDT = datetime.datetime.now()
        print("Unexpected MQTT disconnection! mqttCam" + str(userdata) + currentDT.strftime("  ... %Y-%m-%d %H:%M:%S"))
    sendOne[camn].value=False


def on_connect(client, camn, flags, rc):
    client.subscribe("sendOne/"+str(camn), qos=0)
    #print("mqttCam" + str(camn) + " connected")


def on_message(client, camn, msg):
    global sendOne
    sendOne[camn].value=True
    

# *** start camera reading threads
o = list()
print("[INFO] starting " + str(Nrtsp) + " RTSP Camera Sampling processes ...")
for i in range(Nrtsp):
    o.append(mp.Process(target=rtsp_process, args=(i, rtspURL[i])))
    o[i].start()

while threadsRunning.value < Nrtsp:
    time.sleep(0.5)
currentDT = datetime.datetime.now()
print("[INFO] All " + str(Nrtsp) + " RTSP Camera Sampling processes are running "  + currentDT.strftime("  ... %Y-%m-%d %H:%M:%S"))


while not QUIT:
  try:
    time.sleep(2.0)
  except KeyboardInterrupt:
    QUIT = True
    break
  except Exception as e:
    print('EXCEPTION! ' + str(e))
    print(traceback.format_exc())
    QUIT = True
    break
    
    
for i in range(Nrtsp):
    o[i].join()
print("[INFO] All Camera processes have exited ...")
currentDT = datetime.datetime.now()
print("Program Exit." + currentDT.strftime("  %Y-%m-%d %H:%M:%S"))
print("")
print("")

           

