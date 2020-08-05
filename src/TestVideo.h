#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeaturePoint.h"
#include "FeatureArea.h"
#include "Similarity.h"

using namespace cv;
using namespace std;

/***************************************************************
use cmpVideo: give extra path as parameter
origin picture: pic_path
tourist pic(video): video_path

data_save_path: txt
video_save_path: result video(frame no, result, average time)
***************************************************************/

class TestVideo {

public:
	int pic_no;
	string pic_path;
	int video_no;
	string video_path;
	string data_save_path;
	string video_save_path;

	double video_total_time;

public:
	void cmpVideo_dir(string picp, string videop, string savep);
	void cmpVideo_1v1();

};