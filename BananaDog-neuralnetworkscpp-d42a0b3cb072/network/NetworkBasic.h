#ifndef NETWORKBASIC_H_
#define NETWORKBASIC_H_

#include <armadillo>

#define e 2.7182818284590452353

class NetworkBasic
{
public:
	static double sigmoid(double z);
	static arma::vec sigmoid(arma::vec z);
	static arma::rowvec sigmoid(arma::rowvec z);

	static double sigmoidDerivative(double z);
	static arma::vec sigmoidDerivative(arma::vec z);
};

#endif
