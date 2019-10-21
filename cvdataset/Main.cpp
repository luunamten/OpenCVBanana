#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <cstdlib>
#include <cstring>
#include "BananaWorker.hpp"
#define DRAW_TEXT_COLOR 0,0,0
#define KNN_K 10

cv::Ptr<cv::ml::KNearest> trainKNN();
void genSizeDataSet(const char *path, const char* des);
void getDataFromFile(const char *path, cv::Mat &vectors, cv::Mat &labels);
void recognize(const char *path);
void showBananaType(const char *typeName, cv::Mat &mat);
void recognizeLargeSet();

//A5204

int main(int arg, char *args[]) {
	/*if (arg > 1) {
		if (!std::strcmp(args[1], "k")) {			
			trainKNN();
		}
		else if (!std::strcmp(args[1], "g")) {
			genSizeDataSet(args[2], args[3]);
		}
		else {
			
		}
	}*/
	recognizeLargeSet();
	std::cin.get();
	return 0;
}

void recognizeLargeSet() {
	namespace fx = std::experimental::filesystem;
	const char* path = "../banana_test";
	if (!fx::is_directory(path)) {
		throw std::invalid_argument("../banana_test folder not found.");
	}
	std::string imgDir = "../test_result/";
	for (auto& dir : fx::directory_iterator(path)) {
		const fx::path& imgPath = dir.path();
		if (fx::is_directory(imgPath)) {
			continue;
		}
		std::string filePath = imgDir;
		filePath.append(imgPath.filename().string());
		bn::BananaWorker worker;
		cv::Mat out = worker.open(imgPath.string());
		std::vector<int> params;
		params.push_back(cv::IMWRITE_JPEG_QUALITY);
		params.push_back(100);
		params.push_back(cv::IMWRITE_JPEG_CHROMA_QUALITY);
		params.push_back(100);
		params.push_back(cv::IMWRITE_JPEG_LUMA_QUALITY);
		params.push_back(100);
		cv::imwrite(filePath, out, params);
		std::cout << filePath << std::endl;
	}
}

void showBananaType(const char* typeName, cv::Mat& mat) {
	cv::putText(mat, typeName,
		cv::Point(mat.cols / 2 - 80, 40), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(DRAW_TEXT_COLOR), 2);
	cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
	cv::imshow("window", mat);
}

void recognize(const char* path) {
	namespace fx = std::experimental::filesystem;
	cv::Ptr<cv::ml::KNearest> knnPtr;
	if (fx::exists("../data/KNN.knn")) {
		knnPtr = cv::ml::KNearest::load<cv::ml::KNearest>("../data/KNN.knn");
	}
	else {
		knnPtr = trainKNN();
	}
	bn::BananaWorker worker;
	cv::Mat mat = worker.open(path);
	bn::Size bananaSize = worker.getSize();
	cv::Mat vector(1, 3, CV_32F);
	vector.at<float>(0, 0) = bananaSize.L1;
	vector.at<float>(0, 1) = bananaSize.L2;
	vector.at<float>(0, 2) = bananaSize.H;
	float predictBanana = knnPtr->predict(vector);
	if (predictBanana <= 1) {
		//chuoi cau
		showBananaType("Chuoi Cau", mat);
	}
	else if (predictBanana <= 2) {
		//chuoi gia
		showBananaType("Chuoi Gia", mat);
	}
	else if (predictBanana <= 3) {
		//chuoi su
		showBananaType("Chuoi Su", mat);
	}
	cv::waitKey(0);
}

cv::Ptr<cv::ml::KNearest> trainKNN() {
	namespace fx = std::experimental::filesystem;
	cv::Mat vectors;
	cv::Mat labels;
	cv::Ptr<cv::ml::KNearest> knnPtr{
		cv::ml::KNearest::create()
	};
	getDataFromFile("../data/data.dat", vectors, labels);
	cv::Ptr<cv::ml::TrainData> trainPtr{
		cv::ml::TrainData::create(vectors, cv::ml::SampleTypes::ROW_SAMPLE, labels)
	};
	knnPtr->setIsClassifier(true);
	knnPtr->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
	knnPtr->setDefaultK(KNN_K);
	knnPtr->train(trainPtr);
	knnPtr->save("../data/KNN.knn");
	return knnPtr;
}

void getDataFromFile(const char* fileName, cv::Mat& vectors, cv::Mat& labels) {
	std::ifstream fileStream{ fileName };
	if (!fileStream.is_open()) {
		throw std::invalid_argument("getDataFromFile() can't open the file.");
	}
	std::ifstream countVectorFile{ std::string(fileName) + ".size" };
	if (!countVectorFile.is_open()) {
		throw std::invalid_argument("getDataFromFile() can't open the file.");
	}
	//read header
	int numOfVectors;
	int vectorIndex = 0;
	{
		std::string header;
		std::getline(countVectorFile, header);
		countVectorFile.close();
		std::stringstream strStream{ header };
		strStream >> numOfVectors;
		vectors.create(numOfVectors, 3, CV_32F);
		labels.create(numOfVectors, 1, CV_32S);
	}
	while (!fileStream.eof()) {
		std::string lineStr;
		std::stringstream vectorStrStream;
		std::stringstream labelStrStream;
		//get vector
		std::getline(fileStream, lineStr);
		if (lineStr == "") {
			break;
		}
		vectorStrStream << lineStr;
		vectorStrStream >> vectors.at<float>(vectorIndex, 0)
			>> vectors.at<float>(vectorIndex, 1)
			>> vectors.at<float>(vectorIndex, 2);
		//get label
		std::getline(fileStream, lineStr);
		labelStrStream << lineStr;
		labelStrStream >> labels.at<int>(vectorIndex, 0);
		vectorIndex++;
	}
	fileStream.close();
}

void genSizeDataSet(const char* path, const char* des) {
	namespace fx = std::experimental::filesystem;
	if (!fx::is_directory(path)) {
		throw std::invalid_argument("genSizeDataSet() path isn't a directory.");
	}
	fx::directory_iterator rootDirIterator = fx::directory_iterator(path);
	std::size_t numFolders = std::distance(fx::directory_iterator(path), fx::directory_iterator{});
	std::string newPath = des;
	newPath += "/";
	fx::create_directories(newPath); //directory storing data.csv
	newPath += "data.dat";
	std::ofstream fileStream{ newPath, std::ios::out | std::ios::trunc };
	std::size_t currentType = 1;
	std::size_t countVectors = 0;
	for (auto& dirEntry : rootDirIterator) {
		if (!fx::is_directory(dirEntry)) {
			continue;
		}
		fx::directory_iterator fileIterator = fx::directory_iterator(dirEntry);
		std::string fileName = dirEntry.path().filename().string();
		for (auto& fileEntry : fileIterator) {
			if (!fx::is_regular_file(fileEntry)) {
				continue;
			}
			std::string filePathStr = fileEntry.path().string();
			//banana analyzing
			bn::BananaWorker worker;
			worker.open(filePathStr);
			bn::Size size = worker.getSize();
			fileStream << size.L1 << " " << size.L2 << " " << size.H << std::endl;
			fileStream << currentType << std::endl;
			std::cout << filePathStr << std::endl;
			countVectors++;
		}
		currentType++;
	}
	std::ofstream countVectorFile{ newPath + ".size" };
	countVectorFile << countVectors;
	countVectorFile.close();
	fileStream.close();
}