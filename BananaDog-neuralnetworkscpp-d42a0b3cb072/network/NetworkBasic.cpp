#include "NetworkBasic.h"

double NetworkBasic::sigmoid(double z)
{
	return 1.0/(1.0+pow(e,(-1)*z));
}

arma::vec NetworkBasic::sigmoid(arma::vec z)
{
	arma::vec ones = z;
	ones.ones();		
	return arma::pow((ones+(arma::exp((-1)*z))),-1);
}

arma::rowvec NetworkBasic::sigmoid(arma::rowvec z)
{
	arma::rowvec ones = z;
	ones.ones();		
	return arma::pow((ones+(arma::exp((-1)*z))),-1);
}

double NetworkBasic::sigmoidDerivative(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

arma::vec NetworkBasic::sigmoidDerivative(arma::vec z)
{
	return sigmoid(z)%(1-sigmoid(z));
}
