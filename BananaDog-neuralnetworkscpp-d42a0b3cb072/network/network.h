/*
 * network.h
 *
 *  Created on: Jan 29, 2017
 *      Author: Vaughn McGill-Adami
 */
#ifndef NETWORK_H_
#define NETWORK_H_

#include <armadillo>
#include <iostream>
#include <cmath>

#include "../miscellaneous/Random.h"
#include "NetworkBasic.h"
#include "Costs.h"

struct matpair
{
        arma::mat z;
        arma::mat a;
};

struct cubeandmat
{
        arma::cube w;
        arma::mat b;
};

class Network: public Random, public NetworkBasic
{
public:

	unsigned int num_layers;
	arma::Col<unsigned int> sizes;
	arma::mat biases;
	arma::cube weights;

	//Functions
	Network(arma::Col<unsigned int> imput_sizes);

	matpair Feedfoward(arma::vec a);

	cubeandmat backprop(arma::vec x, arma::vec y, arma::vec (*costdelta)(arma::vec z, arma::vec a, arma::vec y));

	void update_mini_batch(arma::cube mini_batch, double eta, arma::vec (*costdelta)(arma::vec z, arma::vec a, arma::vec y), double reg, unsigned int numtrain);

	void SGD(arma::cube batch,
		 unsigned int epochs,
		 unsigned int minibatch_size,
		 double eta,
		 arma::vec (*costdelta)(arma::vec z, arma::vec a, arma::vec y),
		 double reg,
		 arma::cube test,
		 bool logging = false);

	double evaluate(arma::cube test);
	double cost(arma::cube batch,double (*costfunction)(arma::vec a, arma::vec y));
};



#endif /* NETWORK_H_ */
