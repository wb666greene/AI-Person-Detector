// modified to use OpenCV-4.1.0-openvino
// requires: source useOpenVINO.sh be run first to set paths
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>

using namespace std;

#define NUMOF_STOREABLE_MAPS 3

class FishEyeWindow {
private:
	int srcW_, srcH_, destW_, destH_;
	float al_, be_, th_, R_, zoom_;
	vector<cv::Mat *> mapXs_, mapYs_;
public:
	
    FishEyeWindow(int srcW, int srcH, int destW, int destH)
	: srcW_(srcW), srcH_(srcH), destW_(destW), destH_(destH),
	  al_(0), be_(0), th_(0), R_(srcW / 2.0), zoom_(1.0),
	  mapXs_(NUMOF_STOREABLE_MAPS, NULL), mapYs_(NUMOF_STOREABLE_MAPS, NULL) {}

	~FishEyeWindow() {
		array<vector<cv::Mat *> *, 2> maps = {&mapXs_, &mapYs_};
		for (int i = 0; i < maps.size(); i++) {
			for (vector<cv::Mat *>::iterator it = maps[i]->begin(); it != maps[i]->end(); it++) {
				delete *it;
			}
		}
	}

	void buildMap(float alpha, float beta, float theta, float zoom, int idx = 0) {
		assert(0 <= idx && idx < NUMOF_STOREABLE_MAPS);
		cv::Mat *mapX = new cv::Mat(destH_, destW_, CV_32FC1);
		cv::Mat *mapY = new cv::Mat(destH_, destW_, CV_32FC1);
		// # Set the angle parameters
		al_ = alpha;
		be_ = beta;
		th_ = theta;
		//R_ = R(R, R_)[R == None]
		zoom_ = zoom;
		// # Build the fisheye mapping
		float al = al_ / 180.0;
		float be = be_ / 180.0;
        float th = th_ / 180.0;
        float A = cos(th) * cos(al) - sin(th) * sin(al) * cos(be);
		float B = sin(th) * cos(al) + cos(th) * sin(al) * cos(be);
        float C = cos(th) * sin(al) + sin(th) * cos(al) * cos(be);
        float D = sin(th) * sin(al) - cos(th) * cos(al) * cos(be);
		float mR = zoom_ * R_;
		float mR2 = mR * mR;
		float mRsinBesinAl = mR * sin(be) * sin(al);
		float mRsinBecosAl = mR * sin(be) * cos(al);
		int centerV = int(destH_ / 2.0);
		int centerU = int(destW_ / 2.0);
		float centerY = srcH_ / 2.0;
		float centerX = srcW_ / 2.0;
		// # Fill in the map
		for (int absV = 0; absV < destH_; absV++) {
			float v = absV - centerV;
			float vv = v * v;
			for (int absU = 0; absU < destW_; absU++) {
				float u = absU - centerU;
				float uu = u * u;
				float upperX = R_ * (u * A - v * B + mRsinBesinAl);
				float lowerX = sqrt(uu + vv + mR2);
				float upperY = R_ * (u * C - v * D - mRsinBecosAl);
				float lowerY = lowerX;
				float x = upperX / lowerX + centerX;
				float y = upperY / lowerY + centerY;
				int _v = centerV <= v ? v : v + centerV;
				int _u = centerU <= u ? u : u + centerU;
				mapX->at<float>(_v, _u) = x;
				mapY->at<float>(_v, _u) = y;
			}
		}
		// # Append as new map
		if (mapXs_[idx] != NULL) delete mapXs_[idx];
		if (mapYs_[idx] != NULL) delete mapYs_[idx];
		mapXs_[idx] = mapX;
		mapYs_[idx] = mapY;
	}

	void getImage(cv::Mat &src, cv::Mat &dest, int idx) {
		cv::remap(src, dest, *mapXs_[idx], *mapYs_[idx],
				  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
};

static void errorExit(const char *format, const char *arg)
{
    fprintf(stderr, "Error: ");
    fprintf(stderr, format, arg);
    fprintf(stderr, "\nexit.\n");
    exit(-1);
}

int main(int ac, char *av[])
{
    int dstW=960,dstH=540, srcW,srcH, capfps;
    int saveCnt=0;
    int parmChanged=0;
    char saveFileName[1024];
    FILE *rtspfid = NULL;
    cv::Mat src_img;
    int update=1;       // flag to indicate view mapping needs to be re-calculated
    int picture=0;      // flag to indicate static image file instead of live video
    int key;
    cv::VideoCapture cap;
/*
    float alpha = -270.0;
    float beta = 0.0;
    float theta = 270.0;
    float zoom = 1.0;
*/
    float alpha = 0.0;
    float beta = 90.0;
    float theta = 0.0;
    float zoom = 1.0;
    
    if (ac < 2) {
		printf("Usage: ./%s your-fisheye-image-file or rtsp-stream-URL\n", av[0]);
		return -1;
    }

    const string inFile = av[1];
	cv::namedWindow("Original");
	cv::namedWindow("Result");
	
    if (0 == strstr(av[1], "rtsp")){    // assume image file if not rtsp URL
	    src_img = cv::imread(av[1]);
	    if (!src_img.data) {
		    errorExit("No such image file: %s\n", av[1]);
		    return -1;
        }
        picture=1;
    }else{  // open rtsp stream URL and grab a frame 
        cap.open(inFile); 
        // Check if camera stream opened successfully
        if(!cap.isOpened()){
            errorExit("Error opening video stream or file: %s\n", av[1]);
            return -1;
        }
        //cap >> src_img;

	    if( cap.grab()){
	        cap.retrieve(src_img);
	        printf("srcW %i srcH %i\n", src_img.cols, src_img.rows);
	        cv::imshow("Original", src_img);
	    }else{
	        if(!cap.isOpened()){
	            printf("Camera is not working!\n");
	            errorExit("Error opening video stream or file: %s\n", av[1]);
	        }
	    }
        srcW = cap.get(cv::CAP_PROP_FRAME_WIDTH); 
        srcH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        capfps = cap.get(cv::CAP_PROP_FPS);
        printf("Video Width: %i Height: %i FPS: %i\n", srcW, srcH, capfps);
        if(capfps < 5){
            cap.release();
            picture =1 ;
            printf("FPS is too low for good live view interaction, using only the first frame.\n");
        }else{
            printf("[INFO] Note that rtsp streams typically have a 2-4 second latency.\n");
        }
        printf("\n");
    }
    
	cv::imshow("Original", src_img);

    if( ac >= 4 ){
        sscanf(av[2],"%i",&dstW);
        sscanf(av[3],"%i",&dstH);
    }else{
        printf("Using defaults for dstW dstH alpha beta theta zoom\n");
    }
    if( ac > 4 && ac >= 8 ){
        sscanf(av[4],"%f",&alpha);
        sscanf(av[5],"%f",&beta);
        sscanf(av[6],"%f",&theta);
        sscanf(av[7],"%f",&zoom);
    }else{
        printf("Using defaults for parameters: alpha beta theta zoom\n");
    }
	
	cv::Mat dest_img(dstW, dstH, src_img.type());

    //FishEyeWindow few(src_img.rows, src_img.cols, 320, 320);  // typo in original? src row and columns reversed?
    //FishEyeWindow few(src_img.rows, src_img.cols, dstW, dstH);
    FishEyeWindow few(src_img.cols, src_img.rows, dstW, dstH); // N columns ia image Width
/*    
	printf("Hit followings to move your view:\n"
		   "'r' or 'f' to zoom\n"
		   "'g' or 't' to rotate alpha\n"
		   "'h' or 'y' to rotate beta\n"
		   "'j' or 'u' to rotate theta\n"
		   "'s' to save current view to ./result.png\n"
		   "\n"
		   "Hit ESC to exit.\n");
*/
    printf("Hit following keys to move your view:\n"
        "'z' to Zoom in or 'x' to Zoom out\n"
        "'y' to Yaw right or 't' to Yaw left (rotates alpha)\n"
        "'p' to Pitch up or 'o' to Pitch down (rotates beta)\n"
        "'r' to Roll left or 'e' to Roll right (rotates theta)\n"
        "use upper case letters for smaller increments.\n"
        "'s' or 'S' to save current view to ./result_N.jpg\n"
        "'v' or 'V' to write (or append) current view to ./fisheye.rtsp file\n"
        "\n"
		"Hit ESC to exit.\n"
        "Default Initial view is [alpha=0 beta=90 theta=0 zoom=1.0]\n\n");

	while (true) {
	    if(update){
	        few.buildMap(alpha, beta, theta, zoom);
            printf("[dstW=%i dstH=%i alpha=%.1f beta=%.1f theta=%.1f zoom=%.3f]\n", dstW, dstH, alpha, beta, theta, zoom);
            update=0;
            parmChanged=1;
	    }
		few.getImage(src_img, dest_img, 0);
		cv::imshow("Result", dest_img);
		if(picture) key = cv::waitKey(0); else key = cv::waitKey(30);
        if (key == 27) {
			break;
		}
		switch (key) {
			case 'x':	zoom -= 0.1; update=1; break;
			case 'z':	zoom += 0.1; update=1; break;
			case 'X':	zoom -= 0.01; update=1; break;
			case 'Z':	zoom += 0.01; update=1; break;
			case 'y':	alpha += 10; update=1; break;
			case 't':	alpha -= 10; update=1; break;
			case 'Y':	alpha += 1; update=1; break;
			case 'T':	alpha -= 1; update=1; break;
			case 'p':	beta += 5; update=1; break;
			case 'o':	beta -= 5; update=1; break;
			case 'P':	beta += 0.5; update=1; break;
			case 'O':	beta -= 0.5; update=1; break;
			case 'r':	theta += 10; update=1; break;
			case 'e':	theta -= 10; update=1; break;
			case 'R':	theta += 1; update=1; break;
			case 'E':	theta -= 1; update=1; break;
			case 's':
			case 'S':   sprintf(saveFileName, "./result_%i.jpg", saveCnt++);	
			            cv::imwrite(saveFileName, dest_img); 
			            break;
			case 'v':
			case 'V':   // file format is written in a way to aid reading by Python AI script.
			            if(!parmChanged) break; // trying to avoid accidental duplicate entries
			            if(rtspfid){
			                fprintf(rtspfid,"\n%i %i %.1f %.1f %.1f %.2f", dstW, dstH, alpha, beta, theta, zoom);
			                fflush(rtspfid);
			            }else{
			                rtspfid=fopen("./fisheye.rtsp","a+");
			                if(!rtspfid) rtspfid=fopen("./fisheye.rtsp","w+"); else fprintf(rtspfid,"\n");
			                fprintf(rtspfid,"%s",av[1]);   // camera URL
			                fprintf(rtspfid,"\n%i %i", srcW, srcH); // camera width heigth
			                fprintf(rtspfid,"\n%i %i %.1f %.1f %.1f %.2f", dstW, dstH, alpha, beta, theta, zoom); // write the view param
			                fflush(rtspfid);
			            }
			            parmChanged=0;   	
			            break;
		}
	    if (!picture){
	        if( cap.grab()){
	            cap.retrieve(src_img);
	            cv::imshow("Original", src_img);
	        }else{
	            if(!cap.isOpened()){
	                printf("Camera has died!\n");
	                break;
	            }
	        }
	    }
	}
    if(capfps >= 5) cap.release();
	cv::destroyAllWindows();
    if(rtspfid) fclose(rtspfid);
	return 0;
}
