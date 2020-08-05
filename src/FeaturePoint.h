#ifndef FEATUREPOINT_H  
#define FEATUREPOINT_H 
using namespace cv;
using namespace std;


class FeaturePoint {

public:
	Mat rgbd1, rgbd2;
	double fp_time;

	vector<KeyPoint> Keypoints1;
	vector<KeyPoint> Keypoints2;

	//vector< DMatch > good_matches;
	Mat ShowKeypoints1, ShowKeypoints2;
	Mat ShowMatches;

public:
	int setImage(string p1, string p2);
	int setImage(Mat pic1, Mat pic2);

	void adjVal(double alpha, int beta);

	bool getOrbPicture(Mat rgbd1, Mat rgbd2);
	bool getOrbVideo(Mat rgbd1, Mat rgbd2, vector<KeyPoint> &Keypoints_1, Mat descriptors1);

};

#endif //FEATUREPOINT_H