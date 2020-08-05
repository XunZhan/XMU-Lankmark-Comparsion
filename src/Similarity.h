#ifndef SIMILARITY_H  
#define SIMILARITY_H 

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

using namespace cv;
using namespace std;

struct matchList {
	int c1_index;
	int r1;
	int g1;
	int b1;
	int c2_index;
	int r2;
	int g2;
	int b2;
	int isVaild;
};

class Similarity {

public:
	double simi_time;
	double simiVal;
	int match_cout;
	int vaild_match;
	int point_cout;
	int vaild_point;
	struct matchList ml[MAX_GROUP_NUM];

public:
	void initalMatch();
	int isInclude(int idx, struct cluster cr);
	void matchCluster(FeatureArea fa1, FeatureArea fa2);
	int isAllMatch(FeatureArea fa1, FeatureArea fa2);
	int isSimilar(FeaturePoint fp, FeatureArea fa1, FeatureArea fa2);
	void getClrVal(int idx, FeaturePoint fp, FeatureArea fa1, FeatureArea fa2);
	int clrVaild(int a, int b, int c, int x, int y, int z);
	int posValild(int idx1, int idx2, int side, FeaturePoint fp, FeatureArea fa1, FeatureArea fa2);
};

#endif //SIMILARITY_H  