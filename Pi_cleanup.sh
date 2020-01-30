#!/bin/bash
# edit for path and directory of where detections are stored
BASEDIR=/home/pi
DETECT=detect
# number of days to save detection images and log files
IMGDAYS=7
LOGDAYS=7
/bin/date
echo "Starting cleanup ..."
/usr/bin/find $BASEDIR/$DETECT/ -maxdepth 1 -type d -mtime +$IMGDAYS -exec rm -rf {} \; >/dev/null 2>&1
/usr/bin/find $BASEDIR/$DETECT/ -maxdepth 1 -type f -mtime +$LOGDAYS -exec rm {} \; >/dev/null 2>&1
/bin/date
echo "Finished"

