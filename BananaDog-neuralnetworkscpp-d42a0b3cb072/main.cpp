/*
 * main.cpp
 *
 *  Created on: Feb 4, 2017
 *      Author: Vaughn
 */

#include <armadillo>
#include <vector>
//#include <opencv2/opencv.hpp>
#include <ctime>

#include "./dataprep/LoadData.h"
#include "./dataprep/mnistget.h"
#include "./network/network.h"
#include "./network/Costs.h"

int main()
{
	//arma::cube mnist = LoadData::batch("./data/batch.txt", {4,2,2});
	setbuf(stdout, NULL);

	arma::cube mnist = MnistGet::read_mnist("./data/train-images.idx3-ubyte") + MnistGet::read_labels("./data/train-labels.idx1-ubyte");
	arma::cube test = MnistGet::read_mnist("./data/t10k-images.idx3-ubyte") + MnistGet::read_labels("./data/t10k-labels.idx1-ubyte");

	mnist = mnist(arma::span(0,49999),arma::span::all, arma::span::all);

	std::cout << "Loaded data\n";
	Network net({784,100,10});
	std::cout << "Initialized network\n";
	
	//Performance test	
	time_t tstart, tend;
	tstart = time(0);
        
        std::cout << "Baseline:" << net.evaluate(mnist);

	net.SGD(mnist, 30, 10, 0.01, Costs::CrossEntropydelta, 0, test, true);
	
	matpair output = net.Feedfoward(mnist(arma::span(0),arma::span::all,arma::span::all));

	arma::rowvec a = output.a.row(a.n_rows-1);
	
	a.print("activations:");
	/*
	while(true)
	{
		char b;
		net.update_mini_batch(mnist,1,Costs::CrossEntropydelta,0,4);
		std::cout << "q to quit\n";
		std::cin >> b;
		std::pair<arma::mat, arma::mat>temp = net.Feedfoward({1,1});
        	temp.second.print("Activations:");
        	temp = net.Feedfoward({1,0});
        	temp.second.print("Activations:");
        	temp = net.Feedfoward({0,1});
        	temp.second.print("Activations:");
        	temp = net.Feedfoward({0,0});
        	temp.second.print("Activations:");
		if(b == 'q')
		{
			break;
		}
	}
	*/
	tend = time(0);

        std::cout << "Finished Training:\n" << difftime(tend, tstart) << " seconds\n";
}
