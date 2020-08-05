#ifndef FEATUREAREA_H  
#define FEATUREAREA_H 
using namespace cv;
using namespace std;


#define TOP 0.2
#define MAX_GROUP_NUM 50
#define MAX_FEATURE_NUM 800

struct cluster {
	int index;
	int isMatched;
	double radius;
	int num;
	vector<int> other;
};

struct rho {
	int index;
	double rval;
};

struct delta {
	int index;
	double dval;
};


class FeatureArea {


public:

	int feature_num, cluster_num;
	double dc;
	double pt_dist[MAX_FEATURE_NUM][MAX_FEATURE_NUM];
	struct cluster cr[MAX_GROUP_NUM];
	double fa_time;

public:

	void initCluster(int n, vector<struct rho> &r, vector<struct delta> &d);
	double getDistance(int index1, int index2);
	bool getPtDist(vector<KeyPoint> kp);

	vector<struct rho> getDensity(vector<KeyPoint>kp, vector<struct rho> &r);
	vector<struct delta> getDist2HighDensity(vector<KeyPoint>kp, vector<struct rho> &r, vector<struct delta> &d);

	static bool rcomp(const struct rho &a, const struct rho &b)
	{
		return (a.rval < b.rval);
	}

	static bool dcomp(const struct delta &a, const struct delta &b)
	{
		return (a.dval < b.dval);
	}

	static bool distcomp(const double &a, const double &b)
	{
		return (a < b);
	}

	void clusterPoint(vector<KeyPoint>kp, bool isNorm, vector<struct rho> &r, vector<struct delta> &d);
	int allTrue(vector<bool> isAdd);
	int findDelta(int ri, int from, int to, vector<struct delta> &d);
	int findNear(int ri, int cur_num);
	void rhoCluster(vector<KeyPoint>kp, vector<struct rho> &r, vector<struct delta> &d);

};

#endif //FEATUREAREA_H