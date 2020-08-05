#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <algorithm>

#include<fstream>
#include<iomanip>

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "FeatureArea.h"

using namespace std;
using namespace cv;

#define DC_RATIO 0.02;

void FeatureArea::initCluster(int n, vector<struct rho> &r, vector<struct delta> &d) {
	fa_time = 0;
	cluster_num = 0;
	feature_num = n;
	//printf("init6 pj_val=%d  pj_add=%d\n", *pj, pj);
	//printf("feature_num=%d\n", n);
	for (int i = 0; i < feature_num; i++) {
		r[i].index = i;
		r[i].rval = 0;

		d[i].index = i;
		d[i].dval = 0;
		//printf("init pj_val=%d  pj_add=%d\n", *pj, pj);
	}

	for (int i = 0; i < MAX_GROUP_NUM; i++) {
		cr[i].isMatched = false;
		cr[i].radius = 0;
	}
	//printf("init7 pj_val=%d  pj_add=%d\n", *pj, pj);
}


double FeatureArea::getDistance(int index1, int index2) {
	if (index1 <= index2)
		return pt_dist[index1][index2];
	
	return pt_dist[index2][index1];
}

//in this function dc value is set
bool FeatureArea::getPtDist(vector<KeyPoint> kp) {
	vector<double> dtmp;
	if (kp.size() > MAX_FEATURE_NUM) {
		cout << "kp.size() > MAX_FEATURE_NUM(" << MAX_FEATURE_NUM << ")" << endl;
		system("pause");
		exit(-1);
	}
	for (int i = 0; i < feature_num; i++) {
		for (int j = 0; j < feature_num; j++) {
			if (i < j) {
				pt_dist[i][j] = pow(kp[i].pt.x - kp[j].pt.x, 2) + pow(kp[i].pt.y - kp[j].pt.y, 2);
				pt_dist[i][j] = pow(pt_dist[i][j], 0.5);
				//only when i!=j the distance is vaild
				dtmp.push_back(pt_dist[i][j]);
			}
			else if (i == j)
				pt_dist[i][j] = 0;
		}
	}
	if (dtmp.empty())
	{
		vector<double>().swap(dtmp);
		return false;
	}
	sort(dtmp.begin(), dtmp.end(), distcomp);
	int index = kp.size()*(kp.size() - 1)*DC_RATIO - 1;
	if (index < 0)
		index = 0;
	dc = dtmp[index];

	vector<double>().swap(dtmp);
	return true;
}


vector<struct rho> FeatureArea::getDensity(vector<KeyPoint>kp, vector<struct rho> &r) {

	for (int i = 0; i < feature_num - 1; i++) {
		for (int j = 0; j < feature_num; j++) {
			if (getDistance(i, j) <= dc) {
				++r[i].rval;
				++r[j].rval;
			}
		}
	}
	/********************************************
	ofstream ofile;
	ofile.open("D://Task//Result//rho.txt");
	if (!ofile.is_open()) {
	cout << "1file open error" << endl;
	system("pause");
	exit(-1);
	}

	for (int i = 0; i < n; i++) {
	ofile << rho[i] << endl;
	}

	ofile.close();
	*******************************************/

	return r;
}


vector<struct delta> FeatureArea::getDist2HighDensity(vector<KeyPoint>kp, vector<struct rho> &r, vector<struct delta> &d) {
	for (int i = 0; i < feature_num; i++) {
		double dtmp = 0;
		bool flag = false;
		for (int j = 0; j < feature_num; j++) {
			if (i == j)
				continue;
			if (r[j].rval > r[i].rval) {
				double tmp = getDistance(i, j);
				if (!flag) {
					dtmp = tmp;
					flag = true;
				}
				else {
					if (tmp < dtmp)
						dtmp = tmp;
				}
			}

			if (!flag) {
				for (int k = 0; k < feature_num; k++) {
					double tmp = getDistance(i, k);
					if (tmp > dtmp)
						dtmp = tmp;
				}
			}

			d[i].dval = dtmp;
		}
	}
	/********************************************
	ofstream ofile;
	ofile.open("D://Task//Result//delta.txt");
	if (!ofile.is_open()) {
	cout << "2file open error" << endl;
	system("pause");
	exit(-1);
	}

	for (int i = 0; i < n; i++) {
	ofile << delta[i] << endl;
	}

	ofile.close();
	*******************************************/
	return d;
}


void FeatureArea::clusterPoint(vector<KeyPoint>kp, bool isNorm, vector<struct rho> &r, vector<struct delta> &d) {
	if (isNorm == true) {
		//Normalization
		if (r.size() != d.size()) {
			cout << "rho delta error!\n" << endl;
			system("pause");
			exit(-1);
		}

		std::sort(r.begin(), r.end(), rcomp);
		std::sort(d.begin(), d.end(), dcomp);

		for (int i = 0; i < feature_num; i++) {
			r[i].rval = (r[i].rval - r[0].rval) / (r[feature_num - 1].rval - r[0].rval);
			d[i].dval = (d[i].dval - d[0].dval) / (d[feature_num - 1].dval - d[0].dval);
		}
	}
	else {
		//from small to large
		//but when cluster we use from large to small
		std::sort(r.begin(), r.end(), rcomp);
		std::sort(d.begin(), d.end(), dcomp);
	}

	rhoCluster(kp, r, d);
}

int FeatureArea::allTrue(vector<bool> isAdd) {
	int s = (int)isAdd.size();
	for (int i = 0; i < s; i++) {
		if (!isAdd[i])
			return false;
	}
	return true;
}

int FeatureArea::findDelta(int ri, int from, int to, vector<struct delta> &d) {
	for (int j = from; j < to; j++) {
		if (d[j].index == ri)
			return true;
	}
	return false;
}

int FeatureArea::findNear(int ri, int cur_num) {
	int cluster_index = -1;
	double min_dist;
	int flag = false;
	for (int i = 0; i < cur_num; i++) {
		double tmp = getDistance(cr[i].index, ri);
		if (!flag) {
			min_dist = tmp;
			flag = true;
			if (tmp <= dc*1.3) {
				cluster_index = i;
			}
		}
		else {
			if (tmp < min_dist && tmp <= dc) {
				min_dist = tmp;
				cluster_index = i;
			}
		}
	}
	return cluster_index;
}


void FeatureArea::rhoCluster(vector<KeyPoint>kp, vector<struct rho> &r, vector<struct delta> &d) {
	int count = 0;
	int pre_added = 0, cur_added = -1;
	vector<bool> isAdd(feature_num, 0);

	/*********************************
	if is an isolated point
	its delta is among the top largest
	**********************************/
	for (int i = 0; i <= (int)feature_num*TOP; i++)
		if (!isAdd[i] && findDelta(r[i].index, (int)(feature_num*(1 - TOP) - 1), (int)feature_num, d)) {
			isAdd[r[i].index] = true; pre_added++;
		}

	while (!(cur_added == pre_added || allTrue(isAdd))) {
		if (count > MAX_GROUP_NUM) {
			cout << "count > MAX_GROUP_NUM(" << MAX_GROUP_NUM << ")" << endl;
			system("pause");
			exit(-1);
		}
		//cout << "count= " << count << endl;
		cur_added = pre_added;
		for (int i = feature_num - 1; i >= 0; i--) {
			if (!isAdd[r[i].index]) {
				/***********************
				if is there is no group
				************************/
				if (count == 0) {
					count++;
					cr[0].index = r[i].index;
					cr[0].num = 1;
					isAdd[r[i].index] = true; pre_added++;
				}
				else {
					int cluster_index = findNear(r[i].index, count);

					if (cluster_index >= 0) {
						/***********************
						if it can be included in
						a previous cluster
						************************/
						isAdd[r[i].index] = true; pre_added++;
						cr[cluster_index].num++;
						cr[cluster_index].radius += getDistance(cr[cluster_index].index, r[i].index);
						cr[cluster_index].other.push_back(i);
					}
					else {
						/***********************
						if rho and delta are both
						more than other points
						************************/
						if (i >= (int)(feature_num*(1 - TOP*1.5) - 1) && findDelta(r[i].index, (feature_num*(1 - TOP*1.5) - 1), feature_num, d)) {
							isAdd[r[i].index] = true; pre_added++;
							count++;
							cr[count - 1].index = r[i].index;
							cr[count - 1].num = 1;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < count; i++) {
		if (cr[i].num > 1)
			cr[i].radius = 1.0 * cr[i].radius / (cr[i].num - 1);
		else if (cr[i].num == 1) {
			for (int j = i + 1; j < count; j++) {
				cr[j - 1] = cr[j];
			}
			count--;
		}
	}

	cluster_num = count;
	vector<bool>().swap(isAdd);
	/**************************************************

	ofstream ofile1;
	ofstream ofile2;

	ofile1.open("D://Task//Result//norm_rho.txt");
	if (!ofile1.is_open()) {
	cout << "3file open error" << endl;
	system("pause");
	exit(-1);
	}
	for (int i = 0; i < n; i++) {
	ofile1 << rho[i] << endl;
	}
	ofile1.close();


	ofile2.open("D://Task//Result//norm_delta.txt");
	if (!ofile2.is_open()) {
	cout << "4file open error" << endl;
	system("pause");
	exit(-1);
	}
	for (int i = 0; i < n; i++) {
	ofile2 << delta[i] << endl;
	}
	ofile2.close();
	**************************************************/
}
