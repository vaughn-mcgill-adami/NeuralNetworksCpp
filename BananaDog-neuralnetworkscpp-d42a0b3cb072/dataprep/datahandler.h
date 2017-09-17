#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include <vector>
#include <armadillo>

struct mattrial
{
	arma::mat x;
	arma::vec y;
};

struct vectrial
{
	arma::mat x;
	arma::vec y;
};

arma::vec flatten(arma::mat x);

vectrial flatten(mattrial trial);

arma::mat expand(arma::vec x, unsigned int n_cols);

mattrial expand(vectrial trial, unsigned int n_cols);

#endif
