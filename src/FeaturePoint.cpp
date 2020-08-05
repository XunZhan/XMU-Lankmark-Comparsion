#pragma once
#include "stdafx.h"
#include<iostream>
#include<vector>
#include<time.h>

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\highgui\highgui.hpp>
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp" 

#include "FeatureArea.h"
#include "FeaturePoint.h"

using namespace cv;
using namespace std;

int FeaturePoint::setImage(string p1, string p2) {
	fp_time = 0;
	rgbd1 = imread(p1);
	rgbd2 = imread(p2);
	if (rgbd1.empty())
		return 0;
	if (rgbd2.empty())
		return 0;
	return 1;
}

int FeaturePoint::setImage(Mat pic1, Mat pic2) {
	fp_time = 0;
	rgbd1 = pic1;
	rgbd2 = pic2;
	if (rgbd1.empty())
		return 0;
	return 1;
}

bool FeaturePoint::getOrbVideo(Mat rgbd1, Mat rgbd2, vector<KeyPoint> &Keypoints_1, Mat descriptors1) {

	ORB orb;
	Mat descriptors2;
	vector<DMatch> matches(1000);
	vector<DMatch> good_matches;
	//vector<KeyPoint> Keypoints_1(1000);
	//orb(rgbd1, Mat(), Keypoints_1, descriptors1);

	vector<KeyPoint> Keypoints_2(1000);
	orb(rgbd2, Mat(), Keypoints_2, descriptors2);

	if (Keypoints_1.empty() || Keypoints_2.empty()) {
		cout << "Vector Empty!" << endl;
		return false;
	}

	//cout << "Key points of image" << Keypoints.size() << endl;

	BruteForceMatcher< L2<float> > matcher;
	//BruteForceMatcher<HammingLUT> matcher;
	matcher.match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 250;
	//-- Quick calculation of max and min distances between keypoints  
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);



	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
	//-- PS.- radiusMatch can also be used here.  
	//int cnt = 0;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		//origin < 0.7*
		if (matches[i].distance <= 0.6*max_dist)
		{
			good_matches.push_back(matches[i]);
			KeyPoint tmp1;
			tmp1 = Keypoints_1[matches[i].queryIdx];
			Keypoints1.push_back(tmp1);
			KeyPoint tmp2;
			tmp2 = Keypoints_2[matches[i].trainIdx];
			Keypoints2.push_back(tmp2);
		}
	}

	if (good_matches.empty()) {
		cout << "good_matches empty!" << endl;
		return false;
	}

	drawKeypoints(rgbd1, Keypoints1, ShowKeypoints1);
	drawKeypoints(rgbd2, Keypoints2, ShowKeypoints2);
	drawMatches(rgbd1, Keypoints_1, rgbd2, Keypoints_2,
		good_matches, ShowMatches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	/*
	// localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); ++i)
	{
	// get the keypoints from the good matches
	obj.push_back(Keypoints1[good_matches[i].queryIdx].pt);
	scene.push_back(Keypoints2[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, CV_RANSAC);
	// get the corners from the image_1
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(rgbd1.cols, 0);
	obj_corners[2] = cvPoint(rgbd1.cols, rgbd1.rows);
	obj_corners[3] = cvPoint(0, rgbd1.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	// draw lines between the corners (the mapped object in the scene - image_2)
	line(img_matches, scene_corners[0] + Point2f(rgbd1.cols, 0), scene_corners[1] + Point2f(rgbd1.cols, 0), Scalar(255, 0, 0));
	line(img_matches, scene_corners[1] + Point2f(rgbd1.cols, 0), scene_corners[2] + Point2f(rgbd1.cols, 0), Scalar(255, 0, 0));
	line(img_matches, scene_corners[2] + Point2f(rgbd1.cols, 0), scene_corners[3] + Point2f(rgbd1.cols, 0), Scalar(255, 0, 0));
	line(img_matches, scene_corners[3] + Point2f(rgbd1.cols, 0), scene_corners[0] + Point2f(rgbd1.cols, 0), Scalar(255, 0, 0));
	imshow("Match", img_matches);
	waitKey();
	*/

	//good_matches.swap(vector<DMatch>());
	//obj.swap(vector<Point2f>());
	//scene.swap(vector<Point2f>());

	//vector<KeyPoint>().swap(Keypoints_1);
	vector<KeyPoint>().swap(Keypoints_2);
	vector<DMatch>().swap(matches);
	vector<DMatch>().swap(good_matches);

	//imshow("Match", ShowMatches);
	//waitKey();

	return true;
}



void FeaturePoint::adjVal(double alpha, int beta) {
	/* alpha< 控制对比度[1.0-3.0] */
	/* beta< 控制亮度[0,100] */
	if (rgbd1.cols == rgbd2.cols && rgbd1.rows == rgbd2.rows) {
		for (int y = 0; y < rgbd1.rows; y++)
			for (int x = 0; x < rgbd1.cols; x++)
				for (int c = 0; c < 3; c++)
				{
					rgbd1.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*(rgbd1.at<cv::Vec3b>(y, x)[c]) + beta);
					rgbd2.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*(rgbd2.at<cv::Vec3b>(y, x)[c]) + beta);
				}
		return;
	}
	else {
		for (int y = 0; y < rgbd1.rows; y++)
			for (int x = 0; x < rgbd1.cols; x++)
				for (int c = 0; c < 3; c++)
					rgbd1.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*(rgbd1.at<cv::Vec3b>(y, x)[c]) + beta);

		for (int y = 0; y < rgbd2.rows; y++)
			for (int x = 0; x < rgbd2.cols; x++)
				for (int c = 0; c < 3; c++)
					rgbd2.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*(rgbd2.at<cv::Vec3b>(y, x)[c]) + beta);

	}
	return;

}

bool FeaturePoint::getOrbPicture(Mat rgbd1, Mat rgbd2) {

	ORB orb;
	Mat descriptors1, descriptors2;
	vector<DMatch> matches(1000);
	vector<DMatch> good_matches;

	vector<KeyPoint> Keypoints_1(1000);
	orb(rgbd1, Mat(), Keypoints_1, descriptors1);
	vector<KeyPoint> Keypoints_2(1000);
	orb(rgbd2, Mat(), Keypoints_2, descriptors2);

	if (Keypoints_1.empty() || Keypoints_2.empty()) {
		cout << "Vector Empty!" << endl;
		return false;
	}

	//cout << "Key points of image" << Keypoints.size() << endl;

	BruteForceMatcher< L2<float> > matcher;
	//BruteForceMatcher<HammingLUT> matcher;
	matcher.match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 250;
	//-- Quick calculation of max and min distances between keypoints  
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);



	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
	//-- PS.- radiusMatch can also be used here.  
	//int cnt = 0;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		//origin < 0.7*
		if (matches[i].distance <= 0.6*max_dist)
		{
			good_matches.push_back(matches[i]);
			KeyPoint tmp1;
			tmp1 = Keypoints_1[matches[i].queryIdx];
			Keypoints1.push_back(tmp1);
			KeyPoint tmp2;
			tmp2 = Keypoints_2[matches[i].trainIdx];
			Keypoints2.push_back(tmp2);
		}
	}

	if (good_matches.empty()) {
		cout << "good_matches empty!" << endl;
		return false;
	}

	drawKeypoints(rgbd1, Keypoints1, ShowKeypoints1);
	drawKeypoints(rgbd2, Keypoints2, ShowKeypoints2);
	drawMatches(rgbd1, Keypoints1, rgbd2, Keypoints2,
		good_matches, ShowMatches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<KeyPoint>().swap(Keypoints_1);
	vector<KeyPoint>().swap(Keypoints_2);
	vector<DMatch>().swap(matches);
	vector<DMatch>().swap(good_matches);

	//imshow("Match", ShowMatches);
	//waitKey();

	return true;
}