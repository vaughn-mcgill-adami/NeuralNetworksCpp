#ifndef LOADDATA_H_
#define LOADDATA_H_

#include <armadillo>

class LoadData
{
public:
	static arma::cube batch(std::string dir, arma::vec batchsize);
};

#endif
