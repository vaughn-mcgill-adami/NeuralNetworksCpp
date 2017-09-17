#include "datahandler.h"

arma::vec flatten(arma::mat x)
{
        unsigned int n_rows = x.n_rows;
        unsigned int n_cols = x.n_cols;
        
        arma::vec y(n_rows*n_cols);
	y.zeros();

        for(unsigned int i = 0; i < (n_rows); i++){
                for(unsigned int j = 0; j < (n_cols); j++)
                {
                        y(i*(n_cols)+j)= x(i,j);
                }
        }

        return y;
}

vectrial flatten(mattrial trial)
{
	vectrial ret;
	ret.x = flatten(trial.x);
	ret.y = trial.y;
	return ret;
}

arma::mat expand(arma::vec x, unsigned int n_cols)
{
        unsigned int size = x.size();
        
        if(size%n_cols != 0)
        {
                std::cout << "size must be divisable by n_cols";
        }

        unsigned int n_rows = size/n_cols;

        arma::mat y(n_rows,n_cols);
        
        unsigned int i = 0;
        unsigned int j = 0;
        while(i*n_rows+j < size)
        {
                while(j < n_cols)
                {
                        y(i,j) = x(i*n_cols+j);
                        j++;
                }
                i++;
        }
        return y;
}


mattrial expand(vectrial trial, unsigned int n_cols)
{
	mattrial ret;
	ret.x = expand(trial.x, n_cols);
	ret.y = trial.y;
	return ret;
}

