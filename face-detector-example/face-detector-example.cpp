// face-detector-example.cpp : Defines the entry point for the application.
//

#include "face-detector-example.h"

using namespace std;
using namespace cv;

Rect adjRect(Rect r, double invScale) {
	return Rect(
		Point(cvRound(r.x * invScale), cvRound(r.y * invScale)),
		Point(cvRound(((int64) r.x + r.width) * invScale), cvRound(((int64) r.y + r.height) * invScale)));
}

int main()
{
	String windowName = "Face Detector";
	String filenameHaarCascade = "./data/haarcascades/haarcascade_frontalface_default.xml";
	String filenameNestedHaarCascade = "./data/haarcascades/haarcascade_eye.xml";
	VideoCapture vc;
	Mat frame;
	CascadeClassifier cc;
	CascadeClassifier nc;
	vector<Rect> faces;
	vector<Rect> eyes;
	double scale = 0.5;
	double invScale = 1.0 / scale;


	if (!cc.load(filenameHaarCascade)) {
		cerr << "Couldn't find file " << filenameHaarCascade << endl;
		return 1;
	}

	if (!nc.load(filenameNestedHaarCascade)) {
		cerr << "Couldn't find file " << filenameNestedHaarCascade << endl;
	}

	if (!vc.open(0)) {
		cerr << "Error opening camera index 0" << endl;

		return 1;
	}

	if (!vc.isOpened()) {
		cerr << "vc not openend" << endl;
		return 1;
	}

	namedWindow(windowName, WINDOW_AUTOSIZE);

	int64 t1, t2, t3;
	t1 = t2 = t3 = getTickCount();
	double tfq = 1000.0 / getTickFrequency();

	while (true) {
		
		vc >> frame;
		Mat gray, small;
		t1 = getTickCount();
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		resize(gray, small, Size(), scale, scale, INTER_LINEAR);
		//equalizeHist(small, small);
		t2 = getTickCount();
		cc.detectMultiScale(small, faces, 1.5, 3, 0, Size(20, 20));
		for (auto &face : faces) {
			Mat innerSmall;
		
			rectangle(frame, adjRect(face, invScale), Scalar(0,255,0));
			t3 = getTickCount();
			innerSmall = small(face);
			nc.detectMultiScale(small, eyes, 1.1, 3, 0, Size(5, 5));
			for (auto& eye : eyes) {
				rectangle(frame, adjRect(eye, invScale), Scalar(0, 0, 255));
			}
		}

		imshow(windowName, frame);

		cout << "Resize Time      : " << (t2 - t1) * tfq << endl
			<< "Faces Detect Time: " << (t3 - t2) * tfq << endl
			<< "Eyes Detect Time : " << (getTickCount() - t3) * tfq << endl;
		if (waitKey(300) != -1) {
			break;
		}
	}

	cv::destroyWindow(windowName);
	return 0;
}


