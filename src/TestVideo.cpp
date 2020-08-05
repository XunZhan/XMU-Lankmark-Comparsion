#include "stdafx.h"
#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "Similarity.h"
#include "TestPic.h"
#include "TestVideo.h"

using namespace cv;
using namespace std;

#define SPACE 15
#define DIR_NUM 12
#define VERSION 1

using namespace std;
using namespace cv;

struct dir_info {
	string name;
	int num;
};

void initialDir(struct dir_info dir[DIR_NUM]) {
	dir[0].name = "Bridge"; dir[0].num = 7;
	dir[1].name = "FuRong"; dir[1].num = 7;
	dir[2].name = "JianNan"; dir[2].num = 1;
	dir[3].name = "KeYi"; dir[3].num = 12;
	dir[4].name = "Library"; dir[4].num = 5;
	dir[5].name = "LuXun"; dir[5].num = 3;
	dir[6].name = "Noise"; dir[6].num = 1;
	dir[7].name = "ShangXian"; dir[7].num = 5;
	dir[8].name = "ShiMao"; dir[8].num = 5;
	dir[9].name = "SongEn"; dir[9].num = 10;
	dir[10].name = "Tunnel"; dir[10].num = 4;
	dir[11].name = "ZiQin"; dir[11].num = 7;
}

/****************************************************************************************************
txt格式：最后结果：1=TRUE 0=FALSE

帧数6	用时6	有效匹配区域个数6   总共匹配区域个数6   有效匹配点个数6   总共匹配点个数6  最后结果6
*****************************************************************************************************/

void TestVideo::cmpVideo_dir(string picp, string videop, string savep) {
	struct dir_info dir[DIR_NUM];

	initialDir(dir);
	video_total_time = 0;

	//i:video j:pic
	for (int i = 0; i < DIR_NUM; i++)
	{
		if (dir[i].name.compare("Noise") == 0 || dir[i].name.compare("ShiMao") == 0)
		{
			continue;
		}
		for (int j = 0; j < DIR_NUM; j++)
		{
			cout << "now processing >>Pic: " << dir[j].name << " and >>Video: " << dir[i].name << endl;
			if (i == j)
			{
				cout << "continue" << endl;
				continue;
			}
			//TestVideo tv;
			stringstream ss1; string tmp1;
			ss1.clear();
			ss1 << dir[j].num;
			ss1 >> tmp1;
			pic_no = dir[j].num;
			pic_path = picp + dir[j].name + "\\" + tmp1 + ".jpg";


			stringstream ss2; string tmp2;
			ss2.clear();
			video_no = VERSION;
			ss2 << video_no;
			ss2 >> tmp2;

			video_path = videop + dir[i].name + "\\" + tmp2 + ".avi";

			data_save_path = savep + dir[i].name + "\\" + dir[j].name + "_" + tmp1 + "_" + dir[i].name + "_" + tmp2 + ".txt";
			video_save_path = savep + dir[i].name + "\\" + dir[j].name + "_" + tmp1 + "_" + dir[i].name + "_" + tmp2 + ".avi";
			video_total_time = 0;
			cmpVideo_1v1();
		}
	}
	cout << "finish edit" << endl;
	system("pause");
}

void TestVideo::cmpVideo_1v1() {
	ofstream fout;
	fout.open(data_save_path);

	fout << setw(SPACE) << setfill(' ') << left << "frame";
	fout << setprecision(4) << setw(SPACE) << setfill(' ') << left << "time";
	fout << setw(SPACE) << setfill(' ') << left << "valid_area";
	fout << setw(SPACE) << setfill(' ') << left << "total_area";
	fout << setw(SPACE) << setfill(' ') << left << "valid_point";
	fout << setw(SPACE) << setfill(' ') << left << "total_point";
	fout << setw(SPACE) << setfill(' ') << left << "result" << endl;

	VideoCapture vc(video_path);
	if (!vc.isOpened()) {
		cout << "video open error!" << endl;
		system("pause");
		exit(-1);
	}

	VideoWriter vw;
	vw.open(video_save_path, (int)vc.get(CV_CAP_PROP_FOURCC), (double)vc.get(CV_CAP_PROP_FPS) / 3, Size(256 * 2, 256 * 2));
	if (!vw.isOpened()) {
		cout << "video write error!" << endl;
		system("pause");
		exit(-1);
	}

	bool stop = false;
	int fcount = 0;
	int frame_sum = vc.get(CV_CAP_PROP_FRAME_COUNT);

	/*****************************************
	first get origin picture orb feature point
	******************************************/
	ORB orb;
	Mat origin_pic = imread(pic_path);
	vector<KeyPoint> Keypoints_1(1000);
	Mat descriptors1;
	orb(origin_pic, Mat(), Keypoints_1, descriptors1);

	while (!stop) {
		Mat ftmp;
		Mat frame;
		TestPic tp;
		vc >> ftmp;
		fcount++;
		cout << "frame: " << fcount << endl;
		if (fcount == frame_sum) {
			cout << "finish edit video!" << endl;
			break;
		}

		if (ftmp.empty()) {
			cout << "frame empty error!" << endl;
			break;
			//exit(-1);
		}

		resize(ftmp, frame, Size(256, 256));
		int have = tp.OrbVideo(origin_pic, frame, Keypoints_1, descriptors1);

		if (!have) {
			//no good match
			stringstream ss;
			string s;
			ss.clear();
			ss.str("");
			ss << fcount; ss >> s;
			s = "Frame: " + s;
			Mat tmp = tp.mergeCols(tp.fp.rgbd1, tp.fp.rgbd2);
			Mat fin = tp.mergeRows(tmp, tmp);
			putText(fin, s, Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
			putText(fin, "Result: FALSE", Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

			double time = tp.fp.fp_time + tp.fa1.fa_time + tp.fa2.fa_time + tp.simi.simi_time;
			video_total_time += time;
			double ave_time = video_total_time / fcount;

			string str;
			ss.clear();
			ss.str("");
			ss << time; ss >> str;
			str = "Time = " + str + "s";
			putText(fin, str, Point(0, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

			tp.simi.point_cout = tp.fp.Keypoints2.size();
			fout << setw(SPACE) << setfill(' ') << left << fcount;
			fout << setprecision(4) << setw(SPACE) << setfill(' ') << left << time;
			fout << setw(SPACE) << setfill(' ') << left << 0;
			fout << setw(SPACE) << setfill(' ') << left << tp.simi.match_cout;
			fout << setw(SPACE) << setfill(' ') << left << 0;
			fout << setw(SPACE) << setfill(' ') << left << tp.simi.point_cout;
			fout << setw(SPACE) << setfill(' ') << left << 0 << endl;

			vw << fin;
		}
		else {
			if (!tp.getCluster())
			{
				stringstream ss;
				string s;
				ss.clear();
				ss.str("");
				ss << fcount; ss >> s;
				s = "Frame: " + s;
				Mat tmp = tp.mergeCols(tp.fp.rgbd1, tp.fp.rgbd2);
				Mat fin = tp.mergeRows(tmp, tmp);
				putText(fin, s, Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
				putText(fin, "Result: FALSE", Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

				double time = tp.fp.fp_time + tp.fa1.fa_time + tp.fa2.fa_time + tp.simi.simi_time;
				video_total_time += time;
				double ave_time = video_total_time / fcount;

				string str;
				ss.clear();
				ss.str("");
				ss << time; ss >> str;
				str = "Time = " + str + "s";
				putText(fin, str, Point(0, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

				tp.simi.point_cout = tp.fp.Keypoints2.size();
				fout << setw(SPACE) << setfill(' ') << left << fcount;
				fout << setprecision(4) << setw(SPACE) << setfill(' ') << left << time;
				fout << setw(SPACE) << setfill(' ') << left << 0;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.match_cout;
				fout << setw(SPACE) << setfill(' ') << left << 0;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.point_cout;
				fout << setw(SPACE) << setfill(' ') << left << 0 << endl;

				vw << fin;
			}
			else {
				stringstream ss;
				for (int i = 0; i < tp.fa1.cluster_num; i++)
				{
					double x = tp.fp.Keypoints1[tp.fa1.cr[i].index].pt.x - tp.fa1.cr[i].radius;
					double y = tp.fp.Keypoints1[tp.fa1.cr[i].index].pt.y - tp.fa1.cr[i].radius;

					string s1;
					ss.clear();
					ss.str("");
					ss << i; ss >> s1;
					cv::rectangle(tp.fp.ShowKeypoints1, Rect(x, y, tp.fa1.cr[i].radius * 2, tp.fa1.cr[i].radius * 2), Scalar(0, 0, 255));
					putText(tp.fp.ShowKeypoints1, s1, Point((int)tp.fp.Keypoints1[tp.fa1.cr[i].index].pt.x, (int)tp.fp.Keypoints1[tp.fa1.cr[i].index].pt.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);
				}
				for (int i = 0; i < tp.fa2.cluster_num; i++)
				{
					double x = tp.fp.Keypoints2[tp.fa2.cr[i].index].pt.x - tp.fa2.cr[i].radius;
					double y = tp.fp.Keypoints2[tp.fa2.cr[i].index].pt.y - tp.fa2.cr[i].radius;

					string s2;
					ss.clear();
					ss.str("");
					ss << i; ss >> s2;
					cv::rectangle(tp.fp.ShowKeypoints2, Rect(x, y, tp.fa2.cr[i].radius * 2, tp.fa2.cr[i].radius * 2), Scalar(0, 0, 255));
					putText(tp.fp.ShowKeypoints2, s2, Point((int)tp.fp.Keypoints2[tp.fa2.cr[i].index].pt.x, (int)tp.fp.Keypoints2[tp.fa2.cr[i].index].pt.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);
				}

				clock_t start, end;
				start = clock();
				tp.simi.matchCluster(tp.fa1, tp.fa2);
				int ans = tp.simi.isSimilar(tp.fp, tp.fa1, tp.fa2);
				end = clock();
				tp.simi.simi_time += (double)(end - start) / CLOCKS_PER_SEC;

				Mat tmp = tp.mergeCols(tp.fp.ShowKeypoints1, tp.fp.ShowKeypoints2);
				for (int k = 0; k < tp.simi.match_cout; k++) {
					Point2f p1, p2;
					p1.x = tp.fp.Keypoints1[tp.fa1.cr[tp.simi.ml[k].c1_index].index].pt.x;
					p1.y = tp.fp.Keypoints1[tp.fa1.cr[tp.simi.ml[k].c1_index].index].pt.y;
					p2.x = tp.fp.Keypoints2[tp.fa2.cr[tp.simi.ml[k].c2_index].index].pt.x + tp.fp.ShowKeypoints1.cols;
					p2.y = tp.fp.Keypoints2[tp.fa2.cr[tp.simi.ml[k].c2_index].index].pt.y;
					if (tp.simi.ml[k].isVaild)
						line(tmp, p1, p2, Scalar(0, 0, 255));
					else
						line(tmp, p1, p2, Scalar(255, 255, 255));
				}
				Mat fin = tp.mergeRows(tp.fp.ShowMatches, tmp);

				string s;
				ss.clear();
				ss.str("");
				ss << fcount; ss >> s;
				s = "Frame: " + s;
				putText(fin, s, Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
				if (ans)
					putText(fin, "Result: TRUE", Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
				else
					putText(fin, "Result: FALSE", Point(0, 40), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

				double time = tp.fp.fp_time + tp.fa1.fa_time + tp.fa2.fa_time + tp.simi.simi_time;
				video_total_time += time;
				double ave_time = video_total_time / fcount;

				string st;
				ss.clear();
				ss.str("");
				ss << time; ss >> st;
				st = "Time = " + st + "s";
				putText(fin, st, Point(0, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

				tp.simi.point_cout = tp.fp.Keypoints2.size();
				fout << setw(SPACE) << setfill(' ') << left << fcount;
				fout << setprecision(4) << setw(SPACE) << setfill(' ') << left << time;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.vaild_match;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.match_cout;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.vaild_point;
				fout << setw(SPACE) << setfill(' ') << left << tp.simi.point_cout;
				if (ans)
					fout << setw(SPACE) << setfill(' ') << left << 1 << endl;
				else
					fout << setw(SPACE) << setfill(' ') << left << 0 << endl;

				vw << fin;
			}
		}
	}

	fout.close();
	vw.release();
	vc.release();
}
