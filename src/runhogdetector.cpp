#include "hogdetector.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	HOGDetector detector;
	detector.load("model.xml");
	
	bool CAM = false;
	
	for (int i = 0; i < argc; i++) {
		if (strcmp("cam", argv[i]) == 0) {
			CAM = true;
		} 
	}
	
	VideoCapture cap;
	if (CAM) {
		//cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
        //cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
		cap.open(0);
	} else {
		cap.open(argv[1]);
		cout <<"open capture "<<argv[1] << endl;
	}
        
	if (!cap.isOpened())
	{
		cout << "fail to capture " << argv[1] << endl;
		return -1;
	} 
	
	Mat img;
	int nframe = 0;
	int lost = 0;
	while (1)
	{
		cap >> img;
		if (img.empty())
			break;
		nframe++;
		//resize(img,img,Size(480,320));
		double t = (double)cvGetTickCount();
		//vector<Rect> objs = detector.detect(img(Rect(160,120,320,240)));
		vector<Rect> objs = detector.detect(img);
		if (objs.size() > 0) {
			for (int i = 0; i < objs.size(); i++) {
				rectangle(img, objs.at(i) ,Scalar(0, 255, 0), 3);
			}
		} else {
			lost++;
		}
		t = (double)cvGetTickCount() - t;
		double ms =  t / (cvGetTickFrequency() * 1000);
		double fps = 1000/ms;
		cout << ms<< "ms, " <<  fps << "fps," << nframe << "," << lost << "," << lost/(float)nframe << endl;
		imshow("hogdetector", img);
		waitKey(1);
		
	}
}
