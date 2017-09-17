#include "LoadData.h"

arma::cube LoadData::batch(std::string dir, arma::vec batchsize)
{
	//obvious
	std::fstream txt(dir);
	std::string line;

	//store address in string of where we want to substring line
	unsigned int add1;
	unsigned int add2;

	//For loading each element in the CSV into a cube
	unsigned int train;
	unsigned int tokennumber;
	unsigned int elementnum;
	
	//For storing x/y vectors
	std::string segment;
	//For storing the string of each floating point in the x/y vector
	std::string strdat;
	//return cube
	arma::cube ret(batchsize(0),batchsize(1),batchsize(2));
	ret.zeros();

	if(txt.is_open())
	{	
		//extract training examples
		train = 0;
		while(std::getline(txt,line))
		{
			add1 = 0;
			add2 = 0;
			tokennumber = 0;
			//extract x,y vectors
			while(true)
			{	
				if(tokennumber == 0)
				{
					add1 = line.find("{", (add1));
					add2 = line.find("}", (add2));
				}
				else
				{
					add1 = line.find("{", (add1+1));
					add2 = line.find("}", (add2+1));
				}
				segment = line.substr(add1+1, add2-add1-1);

				//extract individual x and y values for neurons
				elementnum = 0;
				std::stringstream strstream;
				strstream << segment;
				while(getline(strstream, strdat, ','))
				{
					const char * strdatpt = strdat.c_str();
					ret(train,tokennumber,elementnum) = atof(strdatpt);
					elementnum++;
				}
				if(add2 == line.find_last_of("}"))
				{
					break;
				}
				tokennumber++;
			}
			train++;
		}
	}
	return ret;
}
