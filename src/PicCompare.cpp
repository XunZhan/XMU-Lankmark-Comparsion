#include "stdafx.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "TestPic.h"
#include "TestVideo.h"

using namespace std;
using namespace cv;

/*******************************************************
if you want to compare all the video in the dir, remeber
to change:
		1. related dir info in file: "TestVideo.cpp"
		3. video no = VERSION in file:  "TestVideo.cpp"
		2. path in main()
********************************************************/


int main()
{
	/*
	TestVideo tv;
	tv.pic_no = 10;
	tv.pic_path = "D:\\Task\\Test_Data\\Test_256\\SongEn\\10.jpg";
	tv.video_no = 1;
	tv.video_path = "D:\\Task\\Test_Data\\Test_Video\\SongEn\\1.avi";
	tv.data_save_path = "D:\\Task\\Result\\temp\\new.txt";
	tv.video_save_path = "D:\\Task\\Result\\temp\\new.avi";

	tv.cmpVideo_1v1();
	*/

	TestVideo tv;
	string picp = "D:\\Task\\Test_Data\\Test_256\\";
	string videop = "D:\\Task\\Test_Data\\Test_Video\\";
	string savep = "D:\\Task\\Result\\Video_Ver3\\";
	tv.cmpVideo_dir(picp, videop, savep);

	/*
	stringstream ss; string str1,str2;
	int i = 213;
	ss << i;
	ss >> str1;
	str1 = "frame" + str1;
	cout << str1 << endl;

	ss.clear();
	ss.str("");
	int j = 999;
	ss << j;
	ss >> str2;
	cout << str2 << endl;
	str2 = "time" + str2;
	cout << str2 << endl;
	*/

	system("pause");
	return 0;
}

