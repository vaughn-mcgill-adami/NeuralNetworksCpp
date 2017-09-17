/*
 * Random.h
 *
 *  Created on: Feb 4, 2017
 *      Adapted from source here: https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
 *      Thanks Phoxis!
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <armadillo>

class Random {
public:
	double randn(double mu, double sigma);
	void randshuffle(arma::cube & x);
};

#endif /* RANDOM_H_ */
