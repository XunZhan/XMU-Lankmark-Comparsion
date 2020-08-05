#pragma once
#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "Similarity.h"

using namespace cv;
using namespace std;

#define DIR_NUM 12

class TestPic {

public:
	string dir_name[DIR_NUM];
	FeaturePoint fp;
	FeatureArea fa1, fa2;
	Similarity simi;
	double total_time;

public:
	void cmpPic_1v1(string origin_path, string compare_path, string save_path);
	int OrbPicture(string origin_path, string compare_path);
	int OrbVideo(Mat origin_pic, Mat frame, vector<KeyPoint> &Keypoints_1, Mat descriptors1);
	bool getCluster();
	Mat mergeRows(Mat A, Mat B);
	Mat mergeCols(Mat A, Mat B);
};