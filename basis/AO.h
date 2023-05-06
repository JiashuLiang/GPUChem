#ifndef AO_H
#define AO_H

#include <iostream>
#include <armadillo>
#include <vector>

class AO
{
    public:
        arma::vec R0;
        arma::Col<unsigned int> lmn;
        arma::vec alpha;
        arma::vec d_coe;
        int len;
        std::string lable;
        AO(arma::vec R0_input, arma::vec alpha_input, arma::vec d_input, arma::Col<unsigned int> lmn_input, const std::string & lable_input);
        ~AO(){}
        void printinfo();
};



#endif // AO_H