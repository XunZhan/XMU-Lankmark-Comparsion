
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <algorithm>

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "Similarity.h"

using namespace std;
using namespace cv;

struct matchList;
/**********************
struct matchList {
int c1_index;
int r1;
int g1;
int b1;
int c2_index;
int r2;
int g2;
int b2;
};
***********************/

#define TRUE 1
#define FALSE 0
#define NUM_RATIO 0.7
#define CLR_RATIO 0.4
#define VPT_RATIO 0.34
#define VCR_RATIO 0.45
#define DIST_RATIO 0.2

void Similarity::initalMatch() {
	simi_time = 0;
	simiVal = 0;
	match_cout = 0;
	vaild_match = 0;
	point_cout = 0;
	vaild_point = 0;
	for (int i = 0; i < MAX_GROUP_NUM; i++) {
		ml[i].c1_index = -1;
		ml[i].r1 = 0;
		ml[i].g1 = 0;
		ml[i].b1 = 0;
		ml[i].c2_index = -1;
		ml[i].r2 = 0;
		ml[i].g2 = 0;
		ml[i].b2 = 0;
		ml[i].isVaild = FALSE;
	}
}

void Similarity::matchCluster(FeatureArea fa1, FeatureArea fa2) {
	initalMatch();
	int pre_match = 0, cur_match = -1;
	while (!(cur_match == pre_match || isAllMatch(fa1, fa2))) {
		cur_match = pre_match;
		for (int i = 0; i < fa1.cluster_num; i++) {
			if (!fa1.cr[i].isMatched) {
				for (int j = 0; j < fa2.cluster_num; j++) {

					if (!fa1.cr[i].isMatched && !fa2.cr[j].isMatched) {
						/****************************************
						if NUM_RATIO*point number in both clusters
						belongs to other, than they are match
						BUT HERE, i consider if one side suits
						the demand, than it matches BECAUSE:
						finding fa2 matches fa1 is difficultic
						****************************************/
						int q2t = 0;
						if (isInclude(fa1.cr[i].index, fa2.cr[j])) {
							q2t++;
						}
						for (int k = 0; k < fa1.cr[i].other.size(); k++) {
							if (isInclude(fa1.cr[i].other[k], fa2.cr[j])) {
								q2t++;
							}
						}

						if (q2t >= (int)NUM_RATIO*fa1.cr[i].num - 1) {
							pre_match++;
							ml[match_cout].c1_index = i;
							ml[match_cout].c2_index = j;
							fa1.cr[i].isMatched = TRUE;
							fa2.cr[j].isMatched = TRUE;
							match_cout++;
						}
					}


				}
			}

		}
	}

	return;
}

int Similarity::isInclude(int idx, struct cluster cr) {
	if (cr.index == idx)
		return TRUE;
	for (int i = 0; i < cr.other.size(); i++) {
		if (cr.other[i] == idx)
			return TRUE;
	}
	return FALSE;
}

int Similarity::isAllMatch(FeatureArea fa1, FeatureArea fa2) {
	/**********************************************
	in two lists, if one of them is all matched
	then we consider all the clusters are matched
	***********************************************/
	int flag1 = TRUE;
	int flag2 = TRUE;

	for (int i = 0; i < fa1.cluster_num; i++) {
		if (fa1.cr[i].isMatched == FALSE) {
			flag1 = FALSE;
			break;
		}
	}

	for (int i = 0; i < fa2.cluster_num; i++) {
		if (fa2.cr[i].isMatched == FALSE)
		{
			flag2 = FALSE;
			break;
		}
	}

	if (flag1 == FALSE && flag2 == FALSE)
		return FALSE;

	return TRUE;
}

int Similarity::isSimilar(FeaturePoint fp, FeatureArea fa1, FeatureArea fa2) {
	vaild_match = match_cout;
	for (int k = 0; k < match_cout; k++) {
		getClrVal(k, fp, fa1, fa2);
		if (clrVaild(ml[k].r1, ml[k].g1, ml[k].b1, ml[k].r2, ml[k].g2, ml[k].b2) && posValild(ml[k].c1_index, ml[k].c2_index, fp.rgbd1.cols, fp, fa1, fa2)) {
			ml[k].isVaild = TRUE;
			vaild_point += fa2.cr[k].num;
		}
		else
			vaild_match--;
	}
	if (match_cout == 0)
		return FALSE;
	if ((vaild_match < match_cout*VCR_RATIO) || (vaild_point < fp.Keypoints2.size()*VPT_RATIO))
		return FALSE;

	return TRUE;
}

void Similarity::getClrVal(int idx, FeaturePoint fp, FeatureArea fa1, FeatureArea fa2) {
	Mat p1 = fp.rgbd1;
	Mat p2 = fp.rgbd2;
	for (int i = 0; i < fa1.cr[ml[idx].c1_index].other.size(); i++) {
		int x, y;
		//make a change here, original:
		//x = (int)fp.Keypoints1[i].pt.x;
		//y = (int)fp.Keypoints1[i].pt.y;
		x = (int)fp.Keypoints1[fa1.cr[ml[idx].c1_index].other[i]].pt.x;
		y = (int)fp.Keypoints1[fa1.cr[ml[idx].c1_index].other[i]].pt.y;
		ml[idx].r1 += p1.at<Vec3b>(x, y)[0];
		ml[idx].g1 += p1.at<Vec3b>(x, y)[1];
		ml[idx].b1 += p1.at<Vec3b>(x, y)[2];
	}
	int x = (int)fp.Keypoints1[fa1.cr[ml[idx].c1_index].index].pt.x;
	int y = (int)fp.Keypoints1[fa1.cr[ml[idx].c1_index].index].pt.y;
	ml[idx].r1 = (ml[idx].r1 + p1.at<Vec3b>(x, y)[0]) / fa1.cr[ml[idx].c1_index].num;
	ml[idx].g1 = (ml[idx].g1 + p1.at<Vec3b>(x, y)[1]) / fa1.cr[ml[idx].c1_index].num;
	ml[idx].b1 = (ml[idx].b1 + p1.at<Vec3b>(x, y)[2]) / fa1.cr[ml[idx].c1_index].num;

	for (int i = 0; i < fa2.cr[ml[idx].c2_index].other.size(); i++) {
		int x, y;
		x = (int)fp.Keypoints2[i].pt.x;
		y = (int)fp.Keypoints2[i].pt.y;
		ml[idx].r2 += p2.at<Vec3b>(x, y)[0];
		ml[idx].g2 += p2.at<Vec3b>(x, y)[1];
		ml[idx].b2 += p2.at<Vec3b>(x, y)[2];
	}

	x = (int)fp.Keypoints2[fa2.cr[ml[idx].c2_index].index].pt.x;
	y = (int)fp.Keypoints2[fa2.cr[ml[idx].c2_index].index].pt.y;
	ml[idx].r2 = (ml[idx].r2 + p2.at<Vec3b>(x, y)[0]) / fa2.cr[ml[idx].c2_index].num;
	ml[idx].g2 = (ml[idx].g2 + p2.at<Vec3b>(x, y)[1]) / fa2.cr[ml[idx].c2_index].num;
	ml[idx].b2 = (ml[idx].b2 + p2.at<Vec3b>(x, y)[2]) / fa2.cr[ml[idx].c2_index].num;
}

int Similarity::clrVaild(int a, int b, int c, int x, int y, int z) {
	int count = 0;
	if (std::abs(a - x) / (double)a <= CLR_RATIO)
		count++;
	if (std::abs(b - y) / (double)b <= CLR_RATIO)
		count++;
	if (std::abs(c - z) / (double)c <= CLR_RATIO)
		count++;

	if (count >= 2)
		return TRUE;

	return FALSE;
}

int Similarity::posValild(int idx1, int idx2, int side, FeaturePoint fp, FeatureArea fa1, FeatureArea fa2) {
	double x1 = fp.Keypoints1[fa1.cr[idx1].index].pt.x;
	double y1 = fp.Keypoints1[fa1.cr[idx1].index].pt.y;
	double x2 = fp.Keypoints2[fa2.cr[idx2].index].pt.x;
	double y2 = fp.Keypoints2[fa2.cr[idx2].index].pt.y;

	double distx = abs(x1 - x2);
	double disty = abs(y1 - y2);

	if (distx > side*DIST_RATIO || disty > side*DIST_RATIO)
		return FALSE;

	return TRUE;
}