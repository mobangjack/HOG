#include <iostream>
#include <opencv2/opencv.hpp>
#include "hogdetector.hpp"

void HOGDetector::load(const char* model)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(model); 

	int DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数  
	cv::Mat supportVectorMat = svm ->getSupportVectors();//支持向量矩阵  
	int supportVectorNum = supportVectorMat.rows;//支持向量的个数  
	std::cout << "support Vector Num   " << supportVectorNum << std::endl;

	cv::Mat alphaMat = cv::Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//alpha向量，长度等于支持向量个数  
	cv::Mat svindex = cv::Mat::zeros(1, supportVectorNum,CV_64F);
	cv::Mat resultMat ;//alpha向量乘以支持向量矩阵的结果  

	double rho = svm ->getDecisionFunction(0, alphaMat, svindex);

	alphaMat.convertTo(alphaMat, CV_32F);
	//计算-(alphaMat * supportVectorMat),结果放到resultMat中  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？  
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
	
	std::vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中  
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子  
	myDetector.push_back(rho);
	std::cout << "Detector Size   " << myDetector.size() << std::endl;
	
	hog.setSVMDetector(myDetector);

}

std::vector<cv::Rect>  HOGDetector::detect(const cv::Mat& img)
{
	std::cout << "Runing the multi-scale object detection." << std::endl;
	std::vector<cv::Rect> found, found_filtered;
	hog.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(0, 0), 1.1, 2);
	std::cout << "Rect NUM    " << found.size() << std::endl;
	for (int i = 0; i < found.size(); i++)
	{
		cv::Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size()) {
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			found_filtered.push_back(r);
		}
	}
	return found_filtered;
}

