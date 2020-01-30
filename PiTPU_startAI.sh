#!/bin/bash
# edit for directory of AI code and model directories
cd /home/pi/AI

export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority

# should be clean shutdown
/usr/bin/pkill -2 -f "TPU.py" > /dev/null 2>&1
/usr/bin/pkill -2 -f "AI_OVmt.py" > /dev/null 2>&1
sleep 5

# but, make sure it goes away before retrying
/usr/bin/pkill -9 -f "TPU.py" > /dev/null 2>&1
/usr/bin/pkill -9 -f "AI_OVmt.py" > /dev/null 2>&1
sleep 1

export PYTHONUNBUFFERED=1
# necessary only if using OpenVINO cv2
source /opt/intel/openvino/bin/setupvars.sh

./TPU.py -d 0 -cam onvif.txt  >> ../detect/`/bin/date +%F`_AI.log 2>&1 &

