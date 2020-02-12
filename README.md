This is the next step in the evolution of: https://github.com/wb666greene/AI_enhanced_video_security
For some examples of this AI in real-world action checkout the wiki:
https://github.com/wb666greene/AI-Person-Detector/wiki/Camera-notes

The major upgrade is using the Coral TPU and MobilenetSSD-v2_coco for the AI.  The Movidius NCS/NCS2 are still supported, but the *.bin and *.xml files for the MobilenetSSD-v2_coco model are too large to upload to GitHub.

The Ai is pure Python3 code and should work on any system that can run a python3 version supported by Google Coral TPU or Intel OpenVINO installers with an OpenCV version capable of decoding h264/h.265 rtsp streams.  If you have cameras capable of delivering "full resolution" Onvif snapshots, are using USB webcams, mjpeg stream cameras (or motioneyOS), or the PiCamera module the h.264/h.265 decoding issue is moot.

#
## Notes from a "virgin" setup of Raspbian Buster Pi3/4, 22JAN2020

Install your OS using the normal instructions.  I'll use a Pi3B+ and Raspbian "Buster" desktop (2019-09-26-raspbian-buster-full.zip) for this example.  IMHO SD cards are cheap, so buy a big enough one to have a "real" system for testing and development, YMMV.  With Buster the same card can be used in a Pi3 or Pi4.

Once you've done the initial boot/setup steps, here are some things I like to do that aren't setup by default. 
I assume you have a Monitor, Keyboard, and Mouse connected.  IMHO its best to go "headless" only after everything is working.  I'll just outline the basic steps, if Google doesn't give you the details raise an "issue" and we'll flesh out the details.  Feel free to skip any you don't like.

# 
#### These easiest to do via menu->Preferences->RaspberryPiConfiguration:
- set a good password and change the hostname!
- turn off the "splash" screen, I like having the boot messages in case something goes wrong.
- activate ssh server, these extra steps make maintaining a "headless" system almost painless:
   - on your ssh client host create a ssh-id
   - copy it to the Pi with: ssh-copy-id pi@your_hostname
  
#   
#### These steps are best done in a terminal window: 
(or via ssh, I like ssh so I can cut and paste from my desktop with better resources for Google searches)

1. turn off screen blanking, while it doesn't matter headless, I hate screen blanking while setting up and debugging:
	-  sudo nano /etc/xdg/lxsession/LXDE-pi/autostart
	-  Edit to add at the end, these two lines:
      @xset s off
      @xset -dpms

2. setup samba file sharing:
	- sudo apt-get install samba samba-common-bin
	- sudo nano /etc/samba/smb.conf
	Edit these sections to match:
>		[global]
> 			workgroup = your_workgroup
>			mangled names = no
>			; follow symlinks to USB drive
>			follow symlinks = yes
>			wide links = yes
>			unix extensions = no
>
>		[homes]
>			comment = Home Directories
>			browseable = yes
>			read only = no
>			writeable = yes
>			create mask = 0775
>			directory mask = 0775

      - create samba password:
       sudo smbpasswd -a pi

3. I find it useful to have the GUI digital clock display seconds to get an idea of the latency between the cameras and computer.
	- Opposite-click the clock and choose "Digital Clock Settings" from the popup menu.
	- Change %R to %R:%S in the dialog "Clock Format" box, click "OK" button.

# 
#### Install node-red:
- sudo apt-get install build-essential
- bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
- node-red-start
- Make it autostart at boot:
      sudo systemctl enable nodered.service
- install a nodejs package that I use in a function node:
      - cd ~/.node-red
      - npm install point-in-polygon
      - cd .. && nano .node-red/settings.js 
      and add:  insidePolygon:require('point-in-polygon') to functionGlobalContext: {} and restart node-red.
- other bits and pieces not installed by default:
      sudo apt-get install mosquitto mosquitto-dev mosquitto-clients espeak-ng
- extra python modules:
      sudo -H pip3 install paho-mqtt imutils

**In general I recommend the Coral TPU over the Movidius NCS/NCS2, but since the Pi3 lacks USB3 it can't really take full advantage of it. Since I have both and the Python code supports both, I'll set up both. ** On the Pi in a terminal (or via ssh login):
# 
#### Install OpenVINO for Raspbian:
- full instructions: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html
   - download OpenVINO: 
      wget https://download.01.org/opencv/2019/openvinotoolkit/R3/l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz
      (Use the "latest" its dynamic and updated periodically typically 2-4 times a year, and an NCS3 is alledgedly on the way for 2020)
- Create an installation directory:
      sudo mkdir -p /opt/intel/openvino
- Unpack the downloaded archive:
      sudo tar -xf  l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz --strip 1 -C /opt/intel/openvino
- install some needed tools:
      sudo apt install cmake
- activate the OpenVINO environment:
      source /opt/intel/openvino/bin/setupvars.sh
- To use the NCS/NCS2 you need to setup the udev "rules":
      (you don't need the add the current user to the users group, user pi is there by default)
      sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
      
      Plug-in an NCS or NCS2 and do dmesg command, should see output something like this:
        [ 3270.107646] usb 1-1.1.2: New USB device found, idVendor=03e7, idProduct=2150, bcdDevice= 0.01
        [ 3270.107664] usb 1-1.1.2: New USB device strings: Mfr=1, Product=2, SerialNumber=3
        [ 3270.107673] usb 1-1.1.2: Product: Movidius MA2X5X
        [ 3270.107683] usb 1-1.1.2: Manufacturer: Movidius Ltd.
        [ 3270.107692] usb 1-1.1.2: SerialNumber: 03e72150
    
      Its worthwhile to follow the "Build and Run Object Detection Sample" section on the Intel instruction site.

-    Optional:  Make the OpenVINO setup happen on every login with: 
      echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
- #### The MobilenetSSD-v2_coco OpenVINO model files are too large for GitHub.  
I don't think these two steps can be run on the Pi.  Here is my model downloader command:

		~/intel/openvino/deployment_tools/tools/model_downloader$ ./downloader.py --name ssd_mobilenet_v2_coco
		
And my model optimizer command (you need to chage the /home/wally for your system):

	./mo_tf.py --input_model /home/wally/ssdv2/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /home/wally/ssdv2/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/wally/ssdv2/pipeline.config --data_type FP16 --log_level DEBUG
	      
   **At this point you now have a nice version of OpenCV with some extra OpenVINO support functions installed,
   EXCEPT the OpenCV 4.1.2-openvino has issues with mp4 (h.264/h.265) decoding, which breaks using rtsp streams!
   The Pi3B+ is not very usable with rtsp streams and the eariler OpenVINO versions that do work don't support the Pi4.**

# 
#### Setup the Coral TPU: https://coral.ai/docs/accelerator/get-started/
- Google has recently setup a Debian repo that makes it really easy!
      echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
      curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
      sudo apt-get update
      sudo apt-get install python3-edgetpu libedgetpu1-max libedgetpu-dev edgetpu-examples
- Install TensorFlow Lite API:
      ( full instructions: https://www.tensorflow.org/lite/guide/python )
      wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
      pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
- Optional, download and run some test code from Google:
      Plug in the TPU.
      mkdir coral && cd coral
      git clone https://github.com/google-coral/tflite.git
      cd tflite/python/examples/classification
      bash install_requirements.sh
      python3 classify_image.py --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
                                --labels models/inat_bird_labels.txt \
                                --input images/parrot.jpg
      
      You should get something like this:
        INFO: Initialized TensorFlow Lite runtime.
        ----INFERENCE TIME----
        Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
        131.2ms
        11.1ms
        11.2ms
        11.1ms
        11.1ms
        -------RESULTS--------
        Ara macao (Scarlet Macaw): 0.76562

**   The OpenVINO version of OpenCV will work if your cameras do Onvif snapshots, or don't trigger the above mentioned h.264/h.265 decoding issues. **
    
- You need OpenCV for my code, install it (note that as of Jan 2020 there are still issues with newer OpenCV versions):
   sudo apt-get install libhdf5-103 libatlas3-base libjasper1 libqtgui4 libqt4-test libqtcore4
   sudo -H pip3 install opencv-contrib-python==4.1.0.25

   Doing the sudo -H pip3 install allows the OpenVINO and PyPi versions of OpenCV to coexist.  Python virtual environments are prefered, but this is easier and adaquate for an IOT appliance.
   I have a netcam with "h.265+" using a Coral TPU and OpenCV-4.1.2-openvino it gets ~3.3 fps, monotonically increasing latency and eventually crashes.  Using pip installed OpenCV-4.1.0 it gets 5 fps (what the camera is set for) and latency is the typical rtsp ~2 seconds.
   
# 
#### At this point you can download and run my Python code. 
-   After download and unpacking, put the folder in /home/pi and rename it to AI, otherwise you'll have to edit all the controller scripts.
     cd /home/pi/AI
     chmod 755 AI_dev.py AI_OVmt.py TPU.py  Pi4TPU.py *.sh
-    There are four variations of the AI code: AI_dev.py, AI_OVmt.py, TPU.py, and Pi4TPU.py

      AI_dev.py has the most options and can run multiple AI inference threads, its mostly for development and testing, it suports PiCamera module.
      
      AI_OVmt.py by only supports NCS/NCS2 and CPU AI (useless on Pi usable on i3-4025 or better), defults to a single Movidius NCS device.
      
      TPU.py supports a single Coral TPU, no Movidius, no CPU.
      
      Pi4TPU.py is TPU.py with the Pi Camera module  -picam option from AI_dev.py added and -ls local save option removed.
      
      TPU.py, Pi4TPU, & AI_OVmt.py are stand-alone single file scripts needing only some pip installed modules.
      
      AI_dev.py also needs all the *_thread.py files in the same directory along with it.
-    Create a file to specify your onvif snapshot URLs or rtsp stream URLs do a test run:
      If using a Coral TPU and local KVM in a terminal window or ssh -X login:
         ./AI_dev.py -cam snapshots.txt -d 1 -nTPU 1 -ls
         -- OR --
         ./TPU.py -cam snapshots.txt -d 1 -ls
      If using a Movidius NCS or NCS2:
         ./AI_dev.py -cam snapshots.txt -d 1 -nNCS 1 -ls
         -- OR --
         ./AI_OVmt.py -cam snapshots.txt -d 1 -ls
      The saved detection images will be in /home/pi/AI/detect/yyyy-mm-dd/
   e) Exit the AI with Ctrl-C in the terminal window for "q" in the openCV display window

# 
#### Need to do some node-red installation. 
- **You can skip this if you just want to use the AI and do your own thing with integration.**
- If not familar with using node-red, start here:  https://nodered.org/docs/tutorials/
and here:  http://www.steves-internet-guide.com/node-red-overview/
and here:  https://notenoughtech.com/home-automation/nodered-home-automation/nodered-for-beginners-2/
Being a "graphical programming environment" its a bit hard to describe using only text.
Another good set of tutorials, especially to help understand the "dashboard" is:  http://noderedguide.com/
If you learn by watching videos this is a good place to start:  https://www.youtube.com/watch?v=3AR432bguOY

- Connect to your Pi at:  http: local.ip.addr:1880  (or localhost:1880 if running the browser on the Pi)
   From the "hamburger" menu dropdown choose: Manage Pallet
   Click the "Install" tab
   In the search modules box, enter:
     - node-red-node-base64 and click the install button that pops up
     - node-red-dashboard and click the install on the one that exactly matches the search string
     - node-red-node-email and click the install on the one that exactly matches the search string
- Open the Pi_AI_Controller-Viewer.json file from the distribution in your text editor and copy the contents to the clipboard
       - From the node-red menu choose: Import
       - paste the clipboard into the dialog that pops up
       - press the Import to "current flow" button
       - press the red Import button
       - position the graphics and and click, then press the red Deploy button next to the menu
       - if you get "successfully deployed" there is still configuration to be done, but it should be a starting point.
- Open a new browser tab and connect to: http: local.ip.addr:1880/ui
      This views the "dashboard" which does the basic functions like setting the notification mode, viewing a camera, etc.
      When headless, you control it from this webpage, or via MQTT messages from your home automation system.  This is only
      a starting point for you, but you can evaluate the AI performance, and with a WiFi connected Cell Phone adjust camera positions.
- Here is a screen shot of how I've modified the Viewer-Controller flow for my use, although I'm still testing/debugging so the watchdog is not wired up. https://github.com/wb666greene/AI-Person-Detector/blob/master/sample/Controller-Viewer.jpg
- Questions about Node-RED are best asked on this thread where the experts are: [Node-RED Forum](https://discourse.nodered.org/t/ai-enhanced-video-security-system-with-node-red-controller-viewer/21622)

- The sample images on the wiki show why I need a "spacial" filter to not alert on people not on my property.  https://github.com/wb666greene/AI-Person-Detector/wiki/Camera-notes  In some regards the AI is "too good".  The filter function in the sample node-red flow has the skeleton of the nodejs code I use.  I use GIMP to get the polygon coordinates that the lower right corner of the detection must be inside of to generate an alert.

#    
#### Now you can run the AI same as in before but leaving off the -l s option.
   Node-red saves the detections which makes it easier to change the paths and add meaningful names for the cameras.
   You can also change -d 1 to -d 0 which will improve performance by skipping the X display of the live images. 
   You can view them one camera at a time in the UI webpage.
   Viewing the UI webpage and modifying the node-red flow works best with a browser running on a different machine.
   
- If using a Coral TPU and local KVM in a terminal window or ssh -X login:
         ./AI_dev.py -cam snapshots.txt -d 1 -nTPU 1
         -- OR --
         ./TPU.py -cam snapshots.txt -d 1
- If using a Movidius NCS or NCS2:
         ./AI_dev.py -cam snapshots.txt -d 1 -nNCS 1
         -- OR --
         ./AI_OVmt.py -cam snapshots.txt -d 1
- Now saved detection images will be in /home/pi/detect/yyyy-mm-dd/
   You can change the node-red flow to meet your needs and redeploy without stopping and restarting the AI which can be a  real time saver when testing.
  The startup scripts used by the "Launch" inject nodes use -d 0 option.

#  
#### Real world advice.
- SD cards are not the most reliable storage, I recommend formatting  a USB stick ext4 and creating a symlink to it
       for the detections, either:
       - If using local save (delete /home/pi/AI/detect that might have got created while testing):
         ln -s /media/pi/FilesystemLabel  /home/pi/AI/detect
       - If using node-red to save detections (delete /home/pi/detect that might have been created while testing):
         ln -s /media/pi/FilesystemLabel  /home/pi/detect
- Once you've tested things and are ready to go "live": 
       - check "Inject once after 0.5 seconds" on the appropiate Launch inject node
       - wire the watchdog timer to the reboot node
       - prepare for "headless" operation with raspi-config:  BootOptions->Desktop/CLI->Console
       - reboot.  The system should now be an "appliance" that simply boots and runs the AI
- For testing you can use a USB webcam by creating a camera.rtsp  file containing
       /dev/video0
       No blank line at the end!
       launch with the option -rtsp camera.rtsp instead of -cam snapshots.txt
- To use the PiCamera module, launch AI_dev.py with the --PiCam option in addition to any others.
       You can mix the PiCmaera module with other cameras, but the Pi3 is pretty useless with rtsp streams other than /dev/video
       I got ~8 fps using the PiCamera module at 1296x972 resolution on a Pi3B
- A great resource to find the rtsp URL for your cameras is: https://security.world/tools/rtsp-finder/ If you get desperate you can try to "reverse engineer" it with wireshark: http://www.iprogrammable.com/2017/11/10/how-to-use-wireshark-to-get-ip-camera-rtsp-url/
- The "Send Test Email" injection node is useful for setting up your gmail account for sending notifications.
       It needs to have "less secure access" enabled or you need to create an "app key" for it.
- I find MMS texts usually arrive faster (to the same phone) than Email with attachment, but I send myself both.  All
       the four major US carriers have MMS text to Email gateways.  The minor carriers are hit and miss,  Cricket does, Ting doesn't,
       for example.  A Google search for your carrier and "Email to MMS Gateway" should get the answer. I'd be thrilled to accept modified node-red flows and wiki instructions for using Telegram or other push notification options.
- When using rtsp cameras, its best to add 2>/dev/null at the end of the launch command as many (most) cameras throw warnings
       when decoded with OpenCV that make the log files large and mostly useless.
- AI_dev.py still supports using the original NCS version 1 SDK.  If you have it installed, setting up OpenVINO doesn't break anything.
  But I don't think you should bother with it, if you haven't already installed it.
       
# 
## Some performance test results:

- **AI_dev.py Using 5 720p netcams with Onvif snaphots.**
   -   - Pi3B+ running Raspbian Stretch:
            - NCS v1 SDK ~6.5 fps
            - 2 NCS v1 SDK ~11.6 fps
            - NCS OpenVINO ~5.9 fps
            - 2 NCS OpenVINO ~9.9 fps
            - NCS2 OpenVINO ~8.3 fps
     
   - - Odroid XU-4 running Ubuntu Mate-16.04:
         - NCS OpenVINO ~8.5 fps
         - 2 NCS OpenVINO ~15.9 fps
         - NCS2 OpenVINO ~15.5 fps
      

- **TPU.py with 1080p (HD, 3 fps) and 4K (UHD, 5 fps) rtsp streams.**
   -   - Pi4B running Raspbian Buster:
           - 4 HD:     ~11.8 fps (basically processing every frame)
           - 5 HD:     ~14.7 fps (basically processing every frame) 
           - 6 HD:     ~15.0 fps, -d 0 (no display) ~16.7 fps
           - 8 HD:    ~11.6 fps, -d 0 ~14.6 fps
           - 1 UHD:  ~4.6 fps (basically processing every frame)
           - 2 UHD:  ~4 fps very high latency makes it useless (Pi4B struggles with decoding 4K streams)

- **TPU.py with Nano 3 fps HD and UHD rtsp streams.**
   -   - Jetson Nano running Ubuntu 18.04:
           - 5 UHD (4K)         :  ~14.6 fps (effectively processing every frame!)
           - 5 UHD 3 HD         :  ~10.3 fps, jumps to ~19.1 fps if -d 0 option used (no live image display)
           - 4 UHD 4 HD         :  ~16.3 fps, ~22.5 fps with -d 0 option
           - 5 UHD 10 HD (1080p):  ~4.4 fps, ~7.6 fps with -d 0 option (totally overloaded, get ~39 fps when running on i7-4500U MiniPC)
