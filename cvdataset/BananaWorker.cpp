#include "BananaWorker.hpp"
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <climits>

using namespace bn;


BananaWorker::BananaWorker() :
	topLeft{0, 0},
	botRight{0, 0},
	firstPoint{0, 0},
	secondPoint{0, 0},
	thirdPoint{0, 0},
	fourthPoint{0, 0},
	fifthPoint{0, 0},
	size{0, 0, 0}
{}

cv::Mat bn::BananaWorker::open(const char* img)
{
	cv::Mat mat = cv::imread(img, 1);
	if (mat.empty()) {
		throw std::invalid_argument("bn::BananaWorker::open() can't open image.");
	}
	cv::Mat tmp;
	mat.copyTo(tmp);
	work(mat);
	return tmp;
}

cv::Mat bn::BananaWorker::open(std::string img)
{
	cv::Mat mat = cv::imread(img, 1);
	if (mat.empty()) {
		throw std::invalid_argument("bn::BananaWorker::open() can't open image.");
	}
	cv::Mat tmp;
	mat.copyTo(tmp);
	work(mat);
	return tmp;
}

std::pair<cv::Mat, cv::Mat> BananaWorker::preprocess(cv::Mat &mat) {
	//noise reduction
	cv::Mat morphkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE
		, cv::Size(MORP_ELM_REDUCE_NOISE_SIZE));
	cv::morphologyEx(mat, mat, cv::MORPH_CLOSE, morphkernel,
		cv::Point(-1, -1), MORP_NUM_ITERATION);
	cv::morphologyEx(mat, mat, cv::MORPH_OPEN, morphkernel,
		cv::Point(-1, -1), MORP_NUM_ITERATION);
	//gray color
	cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	//noise reduction
	//cv::GaussianBlur(mat, mat, cv::Size(BLUR_SIZE), 0);
	cv::blur(mat, mat, cv::Size(BLUR_SIZE));
	cv::Mat binaryImage;
	double thresh = cv::threshold(mat, binaryImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//edge
	cv::Canny(mat, mat, THRESHHOLD_1, THRESHHOLD_2);
	//find bouding box
	findBounding(mat);
	//get bounding box mat
	cv::Mat cutMat = mat(cv::Range(this->topLeft.y, this->botRight.y + 1),
		cv::Range(this->topLeft.x, this->botRight.x + 1));
	//half image for skeletalizing
	int cutMatHalfCols = (this->topLeft.x + this->botRight.x) / 2;
	cv::Mat half = binaryImage(cv::Range(this->topLeft.y, this->botRight.y + 1),
		cv::Range(cutMatHalfCols, this->botRight.x + 1));
	clearCornerNoise(half);
	cv::Mat skel = this->skeletalize(half);
	return std::make_pair(cutMat, skel);
}

void BananaWorker::findFirst(const cv::Mat &mat) {
	int y1 = -1;
	int y2 = -1;
	int max = 0;
	int limit = -1;
	for (int col = 0; col < mat.cols; col++) {
		for (int row = mat.rows - 1; row >= 0; row--) {
			if (mat.at<uchar>(row, col) > 0) {
				limit = col + OFFSET;
				break;
			}
		}
		if (limit > 0) {
			break;
		}
	}
	for (int col = mat.cols / 2 - 1; col > limit; col--) {
		for (int row = mat.rows - 1; row >= 0; row--) {
			uchar color = mat.at<uchar>(row, col);
			if (color > 0) {
				y2 = row;
				if (y1 >= 0) {
					int diff = y1 - y2;
					if (max <= diff) {
						max = diff;
						this->firstPoint.x = col;
						this->firstPoint.y = y2;
					}
				}
				break;
			}
		}
		if (y2 == -1) {
			break;
		}
		y1 = y2;
		y2 = -1;
	}
	if (max > 1) {
		int lastRow = this->firstPoint.y + max;
		int startCol = this->firstPoint.x + 1;
		for (int rowIndex = this->firstPoint.y + 1; rowIndex < lastRow; rowIndex++) {
			for (int colIndex = startCol; ; colIndex++) {
				uchar color = mat.at<uchar>(rowIndex, colIndex);
				if (color > 0) {
					if (this->firstPoint.x < colIndex - 1) {
						this->firstPoint.x = colIndex - 1;
						this->firstPoint.y = rowIndex - 1;
					}
					break;
				}
			}
			
		}
	}
}

void BananaWorker::findSecond(const cv::Mat &mat) {
	int halfNumMatCols = mat.cols / 2;
	for (int col = mat.cols - 1; col > halfNumMatCols; col--) {
		uchar color = 0;
		for (int row = mat.rows - 1; row >= 0; row--) {
			color = mat.at<uchar>(row, col);
			if (color > 0) {
				int savedRow = row;
				do {
					row--;
				} while (mat.at<uchar>(row, col) > 0);
				this->secondPoint.x = col;
				this->secondPoint.y = std::round((savedRow + row + 1) / 2);
				break;
			}
		}
		if (color != 0) {
			break;
		}
	}
}

cv::Point BananaWorker::findDownward(const cv::Mat &mat, int cols) {
	for (int row = 0; row < mat.rows; row++) {
		uchar color = mat.at<uchar>(row, cols);
		if (color > 0) {
			return cv::Point(cols, row);
		}
	}
	return cv::Point(-1, -1);
}

cv::Point BananaWorker::findUpward(const cv::Mat &mat, int cols) {
	for (int row = mat.rows - 1; row > 0; row--) {
		uchar color = mat.at<uchar>(row, cols);
		if (color > 0) {
			return cv::Point(cols, row);
		}
	}
	return cv::Point(-1, -1);
}


void BananaWorker::findThird(const cv::Mat &mat) {
	int halfNumMatColsM1 = mat.cols / 2 - 1;
	double minDistance = std::numeric_limits<double>::max();
	cv::Point leftPoint = this->findDownward(mat, this->firstPoint.x);
	cv::Point curPoint = this->findDownward(mat, this->firstPoint.x + 1);
	cv::Point rightPoint;
	for (int col = this->firstPoint.x + 1; col < halfNumMatColsM1; col++) {
		rightPoint = this->findDownward(mat, col + 1);
		int leftDy = curPoint.y - leftPoint.y;
		int rightDy = curPoint.y - rightPoint.y;

		if (leftDy < 0 || rightDy < 0) {
			int lastRow;
			if (leftDy > rightDy) {
				lastRow = rightPoint.y;
			}
			else {
				lastRow = leftPoint.y;
			}
			for (int row = curPoint.y; row < lastRow; row++) {
				cv::Point hiddenPoint(col, row);
				double distance = cv::norm(firstPoint - hiddenPoint);
				if (distance < minDistance) {
					minDistance = distance;
					this->thirdPoint = hiddenPoint;
				}
			}
			if (mat.at<uchar>(lastRow, col) > 0) {
				cv::Point hiddenPoint(col, lastRow);
				double distance = cv::norm(firstPoint - hiddenPoint);
				if (distance < minDistance) {
					minDistance = distance;
					this->thirdPoint = hiddenPoint;
				}
			}
		}
		else {
			double distance = cv::norm(firstPoint - curPoint);
			if (distance < minDistance) {
				minDistance = distance;
				this->thirdPoint = curPoint;
			}
		}
		leftPoint = curPoint;
		curPoint = rightPoint;
	}
}

cv::Mat BananaWorker::skeletalize(cv::Mat &mat) {
	cv::Mat skel = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
	cv::Mat erode, dilate;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS,
		cv::Size(MORP_ELM_SKEL_SIZE));
	while (true) {
		cv::erode(mat, erode, kernel);
		cv::dilate(erode, dilate, kernel);
		cv::subtract(mat, dilate, dilate);
		cv::bitwise_or(skel, dilate, skel);
		if (cv::countNonZero(erode) == 0)
			break;
		erode.copyTo(mat);
	}
	//this->binaryImage = skel;
	return skel;
}

void BananaWorker::findFourth(int matHalfCols, const cv::Mat &skel) {
	double maxDistance = skel.cols * skel.rows;
	cv::Point topRight{skel.cols - 1, 0};
	for (int col = skel.cols - 1; col >= 0; col--) {
		
		for (int row = 0; row < skel.rows; row++) {
			if (skel.at<uchar>(row, col) > 0) {
				double distance = cv::norm(topRight - cv::Point{col, row});
				if(distance < maxDistance) {
					maxDistance = distance;
					this->fourthPoint.x = col;
					this->fourthPoint.y = row;
				}
				break;
			}
		}
	}
	this->fourthPoint.x += matHalfCols;
}

void BananaWorker::findFifth(const cv::Mat &mat) {
	double maxDistance = 0;
	cv::Point leftPoint = this->findUpward(mat, this->firstPoint.x);
	cv::Point curPoint = this->findUpward(mat, this->firstPoint.x + 1);
	cv::Point rightPoint;
	cv::Vec2i normal;
	double normalLength;
	double c;
	cv::Vec2i dir = this->fourthPoint - this->thirdPoint;
	normal[0] = -dir[1];
	normal[1] = dir[0];
	normalLength = cv::norm(normal);
	c = -normal[0] * this->thirdPoint.x - normal[1] * this->thirdPoint.y;
	for (int col = this->firstPoint.x + 1; col < this->secondPoint.x; col++) {
		rightPoint = this->findUpward(mat, col + 1);
		int leftDy = curPoint.y - leftPoint.y;
		int rightDy = curPoint.y - rightPoint.y;
		if (leftDy > 0 || rightDy > 0) {
			int lastRow;
			if (leftDy > rightDy) {
				lastRow = leftPoint.y;
			}
			else {
				lastRow = rightPoint.y;
			}
			for (int row = curPoint.y; row > lastRow; row--) {
				cv::Point hiddenPoint(col, row);
				double distance = std::abs(normal[0] * hiddenPoint.x + normal[1] * hiddenPoint.y + c) / normalLength;
				if (distance > maxDistance) {
					maxDistance = distance;
					this->fifthPoint = hiddenPoint;
				}
			}
			if (mat.at<uchar>(lastRow, col) > 0) {
				cv::Point hiddenPoint(col, lastRow);
				double distance = std::abs(normal[0] * hiddenPoint.x + normal[1] * hiddenPoint.y + c) / normalLength;
				if (distance > maxDistance) {
					maxDistance = distance;
					this->fifthPoint = hiddenPoint;
				}
			}
		}
		else {
			double distance = std::abs(normal[0] * curPoint.x + normal[1] * curPoint.y + c) / normalLength;
			if (distance > maxDistance) {
				maxDistance = distance;
				this->fifthPoint = curPoint;
			}
		}
		leftPoint = curPoint;
		curPoint = rightPoint;
	}
	this->size.L2 = cv::norm(dir);
	this->size.H = maxDistance;
}

void BananaWorker::findL1(const cv::Mat &mat) {
	cv::Mat tmp;
	mat.copyTo(tmp);
	Fx fx;
	double fifthPointSign;
	{
		cv::Vec2i dir = this->firstPoint - this->secondPoint;
		fx.normal[0] = -dir[1];
		fx.normal[1] = dir[0];
		fx.normal = fx.normal / cv::norm(dir);
		fx.c = -(fx.normal[0] * this->secondPoint.x + fx.normal[1] * this->secondPoint.y);
	}
	fifthPointSign = fx(this->fifthPoint);
	if (fifthPointSign > 0) {
		fifthPointSign = 1;
	}
	else if (fifthPointSign < 0) {
		fifthPointSign = -1;
	}
	else {
		fifthPointSign = 0;
	}
	this->size.L1++;
	findL1Recur(fx, this->firstPoint, tmp, fifthPointSign);
}

void BananaWorker::findL1Recur(const Fx &fx, const cv::Point &center, cv::Mat &mat, double fifthPointSign) {
	for (int row = -1; row <= 1; row++) {
		for (int col = -1; col <= 1; col++) {
			if (col == 0 && row == 0) {
				continue;
			}
			cv::Point p(center.x + col, center.y + row);
			uchar &color = mat.at<uchar>(p.y, p.x); 
			if (color != 0 && color != CONTOUR_COLOR && fx(p) * fifthPointSign >= 0) {
				color = CONTOUR_COLOR;
				this->size.L1++;
				findL1Recur(fx, p, mat, fifthPointSign);
			}
		}
	}
}

void BananaWorker::work(cv::Mat &mat) {
	std::pair<cv::Mat, cv::Mat> pair = this->preprocess(mat);
	//find five point
	this->findFirst(pair.first);
	this->findSecond(pair.first);
	this->findThird(pair.first);
	this->findFourth(pair.first.cols / 2, pair.second);
	this->findFifth(pair.first);
	this->findL1(pair.first);
}

Size BananaWorker::getSize()
{
	return this->size;
}

cv::Mat BananaWorker::findBounding(cv::Mat& cannyMat) {
	cv::Mat tmp;
	cannyMat.copyTo(tmp); //create new Mat for marking
	cv::Point start{ tmp.cols / 2, tmp.rows / 4};
	while (start.y < tmp.rows) {
		uchar& color = tmp.at<uchar>(start);
		if (color > 0) {
			color = CONTOUR_COLOR;
			break;
		}
		start.y++;
	}
	this->topLeft = start;
	this->botRight = start;
	findBoundingRecur(tmp, start);
	this->topLeft.x -= PADDING;
	this->topLeft.y -= PADDING;
	this->botRight.x += PADDING;
	this->botRight.y += PADDING;
	return tmp;
}

void BananaWorker::checkPointIsOutside(cv::Mat& mat, const cv::Point& center) {
	for (int dx = -1; dx <= 1; dx++) {
		for (int dy = -1; dy <= 1; dy++) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			cv::Point point{ center.x + dx, center.y + dy };
			if (!isPointInMat(mat, point) || mat.at<uchar>(point) == 0) {
				mat.at<uchar>(center) = CONTOUR_COLOR;
				findBoundingRecur(mat, center);
				return;
			}
		}
	}
}

void BananaWorker::findBoundingRecur(cv::Mat& mat, const cv::Point& center) {
	if (topLeft.x > center.x) {
		topLeft.x = center.x;
	}
	else if (botRight.x < center.x) {
		botRight.x = center.x;
	}
	if (topLeft.y > center.y) {
		topLeft.y = center.y;
	}
	else if (botRight.y < center.y) {
		botRight.y = center.y;
	}

	for (int dx = -1; dx <= 1; dx++) {
		for (int dy = -1; dy <= 1; dy++) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			cv::Point point{ center.x + dx, center.y + dy };
			if (isPointInMat(mat, point) && mat.at<uchar>(point) > CONTOUR_COLOR) {
				checkPointIsOutside(mat, point);
			}
		}
	}
}

void BananaWorker::clearCornerNoise(cv::Mat& mat) {
	for (int row = 0; row < mat.rows; row++) {
		clearCornerNoiseRecur(mat, cv::Point{mat.cols - 1, row});
	}
	for (int col = 0; col < mat.cols; col++) {
		clearCornerNoiseRecur(mat, cv::Point{ col, 0});
	}
}

void BananaWorker::clearCornerNoiseRecur(cv::Mat& mat, const cv::Point& corner) {
	mat.at<uchar>(corner) = 0;
	for (int dx = -1; dx <= 1; dx++) {
		for (int dy = -1; dy <= 1; dy++) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			cv::Point point{corner.x + dx, corner.y + dy};
			if (isPointInMat(mat, point) && mat.at<uchar>(point) > 0) {
				clearCornerNoiseRecur(mat, point);
			}
		}
	}
}

bool BananaWorker::isPointInMat(const cv::Mat& mat, const cv::Point& point) {
	return point.x >= 0 && point.x < mat.cols && point.y >= 0 && point.y < mat.rows;
}

/*--------------------------FX----------------*/
inline double Fx::operator ()(const cv::Point &p) const {
	return this->normal[0] * p.x + this->normal[1] * p.y + this->c;
}
