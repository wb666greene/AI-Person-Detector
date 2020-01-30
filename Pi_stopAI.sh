#!/bin/bash
# edit for directory of AI code and model directories
cd /home/pi/AI

export DISPLAY=:0
export XAUTHORITY=/home/ai/.Xauthority

# should be clean shutdown
/usr/bin/pkill -2 -f "TPU.py" > /dev/null 2>&1
/usr/bin/pkill -2 -f "AI_OVmt.py" > /dev/null 2>&1
sleep 3

# but, make sure it goes away before retrying
/usr/bin/pkill -9 -f "Pi4TPU.py" > /dev/null 2>&1
/usr/bin/pkill -9 -f "AI_OVmt.py" > /dev/null 2>&1
sleep 1

