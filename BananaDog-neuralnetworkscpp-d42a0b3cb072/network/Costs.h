#ifndef COSTS_H_
#define COSTS_H_

#include <armadillo>
#include "NetworkBasic.h"

class Costs: public NetworkBasic
{
public:
        static double CrossEntropy(arma::vec a, arma::vec y);
	//returns a vector of d(C)/d(z^L)
        static arma::vec CrossEntropydelta(arma::vec z, arma::vec a, arma::vec y);

	static double QuadraticCost(arma::vec a, arma::vec y);
	static arma::vec QuadraticCostdelta(arma::vec z, arma::vec a, arma::vec y);

	static double LogLikelihood(arma::vec a, arma::vec y);
	static arma::vec LogLikelihooddelta(arma::vec z, arma::vec a, arma::vec y);
};

#endif
