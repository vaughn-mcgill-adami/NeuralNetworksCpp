#include "Costs.h"

//This function affects nothing as of update 0.3
double Costs::CrossEntropy(arma::vec a, arma::vec y)
{
	return (-1)*(arma::accu((y%arma::log(a))+((1-y)%arma::log(1-a))));
}

//This function is compatible with Network::SGD
arma::vec Costs::CrossEntropydelta(arma::vec z, arma::vec a, arma::vec y)
{
	return a-y;
}

//This function affects nothing as of update 0.3
double Costs::QuadraticCost(arma::vec a, arma::vec y)
{
	return 0.5*(arma::sum(arma::pow(a-y,2)));
}

//This function is compatible with Network::SGD
arma::vec Costs::QuadraticCostdelta(arma::vec z, arma::vec a, arma::vec y)
{
	return (a-y)%sigmoidDerivative(z);
}

//This function affects nothing as of update 0.3
double Costs::LogLikelihood(arma::vec a, arma::vec y)
{
	return (-1*log(a(y.index_max())));
}

//This function is compatible with Network::SGD, but as of update 0.3 it is can only be used
//as a replacement for Costs::CrossEntropydelta because 0.3 has no softmax output layers
arma::vec Costs::LogLikelihooddelta(arma::vec z, arma::vec a, arma::vec y)
{
	return (a-y);
}
