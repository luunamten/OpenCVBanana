#ifndef M_BANANA_WORKER
#define M_BANANA_WORKER
#include <cmath>
#include <algorithm>
#include <string>
#include <initializer_list>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <limits>
#include <utility>
#define THRESHHOLD_1 40
#define THRESHHOLD_2 THRESHHOLD_1 * 3
#define MORP_ELM_REDUCE_NOISE_SIZE 5,5
#define MORP_NUM_ITERATION 2
#define MORP_ELM_SKEL_SIZE 3,3
#define BLUR_SIZE 3,3
#define OFFSET 1
#define PADDING 5
#define CONTOUR_COLOR 100

namespace bn{
	struct Size;
	struct Fx;
	class BananaWorker;
}

struct bn::Size {
	double L1; //length
	double L2; //ventral straight length
	double H; // arc height
};

struct bn::Fx {
	cv::Vec2d normal;
	double c;
	inline double operator()(const cv::Point &p) const;
};

class bn::BananaWorker
{
private:
	void work(cv::Mat &mat);
	std::pair<cv::Mat, cv::Mat> preprocess(cv::Mat &mat);
	cv::Point findDownward(const cv::Mat &mat, int cols);
	cv::Point findUpward(const cv::Mat &mat, int cols);
	void findFirst(const cv::Mat &mat);
	void findSecond(const cv::Mat &mat);
	void findThird(const cv::Mat &mat);
	void findFourth(int matHalfCols, const cv::Mat &half);
	void findFifth(const cv::Mat &mat);
	cv::Mat skeletalize(cv::Mat &mat);
	void findL1(const cv::Mat &mat);
	void findL1Recur(const Fx &fx, const cv::Point &center, cv::Mat &mat, double fifthPointSign);
	cv::Mat findBounding(cv::Mat& cannyMat);
	void findBoundingRecur(cv::Mat &mat, const cv::Point& center);
	void checkPointIsOutside(cv::Mat &mat, const cv::Point& point);
	void clearCornerNoise(cv::Mat &mat);
	void clearCornerNoiseRecur(cv::Mat &mat, const cv::Point &corner);
	bool isPointInMat(const cv::Mat &mat, const cv::Point &point);
	cv::Point topLeft;
	cv::Point botRight;
	cv::Point firstPoint;
	cv::Point secondPoint;
	cv::Point thirdPoint;
	cv::Point fourthPoint;
	cv::Point fifthPoint;
	bn::Size size;
	cv::Mat img;
public:
	BananaWorker();
	BananaWorker(const BananaWorker &b) = default;
	BananaWorker(BananaWorker &&b) = default;
	BananaWorker& operator=(const BananaWorker &b) = default;
	BananaWorker& operator=(BananaWorker &&b) = default;
	cv::Mat open(const char* img);
	cv::Mat open(std::string img);
	Size getSize();
	~BananaWorker() = default;
};
#endif