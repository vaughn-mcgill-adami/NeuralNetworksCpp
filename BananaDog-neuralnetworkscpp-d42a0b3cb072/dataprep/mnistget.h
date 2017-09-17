#ifndef MNISTGET_H_
#define MNISTGET_H_

#include <armadillo>
#include <iostream>
#include <vector>
#include <string>

#include "datahandler.h"

using namespace std;

class MnistGet
{
public:
	static int reverseInt(int i);
	static arma::cube read_mnist(std::string dir);
	//static std::vector<cv::Mat> read_mnistimg();
	static arma::cube read_labels(std::string dir);
};
#endif
