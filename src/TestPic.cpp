#include "stdafx.h"
#include<iostream>
#include<vector>
#include<time.h>

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "Similarity.h"
#include "TestPic.h"

using namespace cv;
using namespace std;


void TestPic::cmpPic_1v1(string origin_path, string compare_path, string save_path) {
	int f;
	f = OrbPicture(origin_path, compare_path);
	if (!f) {
		//no good match
		Mat tmp = mergeCols(fp.rgbd1, fp.rgbd2);
		Mat fin = mergeRows(tmp ,tmp);
		putText(fin, "FALSE", Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));

		double time = fp.fp_time + fa1.fa_time + fa2.fa_time + simi.simi_time;
		total_time = total_time + time;
		stringstream ss; string s;
		ss << time; ss >> s;
		s = "time = " + s + "s";
		putText(fin, s, Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));
		imwrite(save_path, tmp);
		return;
	}
	//have good matches
	getCluster();
	for (int i = 0; i < fa1.cluster_num; i++)
	{
		double x = fp.Keypoints1[fa1.cr[i].index].pt.x - fa1.cr[i].radius;
		double y = fp.Keypoints1[fa1.cr[i].index].pt.y - fa1.cr[i].radius;
		stringstream ss;
		string str;
		ss << i; ss >> str;
		cv::rectangle(fp.ShowKeypoints1, Rect(x, y, fa1.cr[i].radius * 2, fa1.cr[i].radius * 2), Scalar(0, 0, 255));
		putText(fp.ShowKeypoints1, str, Point((int)fp.Keypoints1[fa1.cr[i].index].pt.x, (int)fp.Keypoints1[fa1.cr[i].index].pt.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);
	}
	for (int i = 0; i < fa2.cluster_num; i++)
	{
		double x = fp.Keypoints2[fa2.cr[i].index].pt.x - fa2.cr[i].radius;
		double y = fp.Keypoints2[fa2.cr[i].index].pt.y - fa2.cr[i].radius;
		stringstream ss;
		string str;
		ss << i; ss >> str;
		cv::rectangle(fp.ShowKeypoints2, Rect(x, y, fa2.cr[i].radius * 2, fa2.cr[i].radius * 2), Scalar(0, 0, 255));
		putText(fp.ShowKeypoints2, str, Point((int)fp.Keypoints2[fa2.cr[i].index].pt.x, (int)fp.Keypoints2[fa2.cr[i].index].pt.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);
	}

	clock_t start, end;
	start = clock();
	simi.matchCluster(fa1, fa2);
	int ans = simi.isSimilar(fp, fa1, fa2);
	end = clock();
	simi.simi_time += (double)(end - start) / CLOCKS_PER_SEC;

	Mat tmp = mergeCols(fp.ShowKeypoints1, fp.ShowKeypoints2);
	for (int k = 0; k < simi.match_cout; k++) {
		Point2f p1, p2;
		p1.x = fp.Keypoints1[fa1.cr[simi.ml[k].c1_index].index].pt.x;
		p1.y = fp.Keypoints1[fa1.cr[simi.ml[k].c1_index].index].pt.y;
		p2.x = fp.Keypoints2[fa2.cr[simi.ml[k].c2_index].index].pt.x + fp.ShowKeypoints1.cols;
		p2.y = fp.Keypoints2[fa2.cr[simi.ml[k].c2_index].index].pt.y;
		if (simi.ml[k].isVaild)
			line(tmp, p1, p2, Scalar(0, 0, 255));
		else
			line(tmp, p1, p2, Scalar(255, 255, 255));
	}
	Mat fin = mergeRows(fp.ShowMatches, tmp);
	if (ans)
		putText(fin, "TRUE", Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));
	else
		putText(fin, "FALSE", Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));

	double time = fp.fp_time + fa1.fa_time + fa2.fa_time + simi.simi_time;
	total_time = time;
	stringstream ss; string s;
	ss << time; ss >> s;
	s = "time = " + s + "s";
	putText(fin, s, Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));
	imwrite(save_path, fin);
	//imshow("test", fin);
	//waitKey();
	return;
}


int TestPic::OrbPicture(string origin_path, string compare_path) {
	clock_t start, end;
	start = clock();

	if (!fp.setImage(origin_path, compare_path)) {
		cout << "Load Image Error!" << endl;
		system("pause");
		exit(-1);
	}

	if (!fp.getOrbPicture(fp.rgbd1, fp.rgbd2))
		return false;

	if (fp.Keypoints1.empty() || fp.Keypoints2.empty()) {
		cout << "Vector Empty!" << endl;
		return false;
	}
	end = clock();

	fp.fp_time += (double)(end - start) / CLOCKS_PER_SEC;
	return true;
}


int TestPic::OrbVideo(Mat origin_pic, Mat frame, vector<KeyPoint> &Keypoints_1, Mat descriptors1) {
	clock_t start, end;
	start = clock();

	Mat pic1 = origin_pic;
	Mat pic2 = frame;


	if (!fp.setImage(pic1, pic2)) {
		cout << "Load Image Error!" << endl;
		system("pause");
		exit(-1);
	}

	if (!fp.getOrbVideo(fp.rgbd1, fp.rgbd2, Keypoints_1, descriptors1))
		return false;

	if (fp.Keypoints1.empty() || fp.Keypoints2.empty()) {
		cout << "Vector Empty!" << endl;
		return false;
	}
	end = clock();

	fp.fp_time += (double)(end - start) / CLOCKS_PER_SEC;
	return true;
}

bool TestPic::getCluster() {
	//Mat tmp = ShowKeypoints;
	//temp1 = ShowKeypoints.size();
	clock_t start, end;

	start = clock();
	vector<struct rho> r1(fp.Keypoints1.size());
	vector<struct delta> d1(fp.Keypoints1.size());
	fa1.initCluster(fp.Keypoints1.size(), r1, d1);
	//in this function dc value is set
	if (!fa1.getPtDist(fp.Keypoints1))
		return false;
	r1 = fa1.getDensity(fp.Keypoints1, r1);
	d1 = fa1.getDist2HighDensity(fp.Keypoints1, r1, d1);
	fa1.clusterPoint(fp.Keypoints1, false, r1, d1);
	end = clock();
	fa1.fa_time += (double)(end - start) / CLOCKS_PER_SEC;

	start = clock();
	vector<struct rho> r2(fp.Keypoints2.size());
	vector<struct delta> d2(fp.Keypoints2.size());
	fa2.initCluster(fp.Keypoints2.size(), r2, d2);
	if (!fa2.getPtDist(fp.Keypoints2))
		return false;
	r2 = fa2.getDensity(fp.Keypoints2, r2);
	d2 = fa2.getDist2HighDensity(fp.Keypoints2, r2, d2);
	fa2.clusterPoint(fp.Keypoints2, false, r2, d2);
	end = clock();
	fa2.fa_time += (double)(end - start) / CLOCKS_PER_SEC;

	return true;

}

Mat TestPic::mergeRows(Mat A, Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	Mat mergedDescriptors(totalRows, A.cols, A.type());
	Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}

Mat TestPic::mergeCols(Mat A, Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalCols = A.cols + B.cols;
	Mat mergedDescriptors(A.rows, totalCols, A.type());
	Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.colRange(A.cols, totalCols);
	B.copyTo(submat);
	return mergedDescriptors;
}