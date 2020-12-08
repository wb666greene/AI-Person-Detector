# requires: source useOpenVINO.sh be run first to set paths
g++ -std=gnu++11 -g -o fisheye_window  fisheye_windowOV.cpp -I/opt/intel/openvino/opencv/include/ -L/opt/intel/openvino/opencv/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_imgcodecs -lopencv_videoio
