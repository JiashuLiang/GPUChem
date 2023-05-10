#ifndef AO_H
#define AO_H

#include <iostream>
#include <armadillo>
#include <vector>

class AO
{
    public:
        arma::vec R0;
        arma::Col<unsigned int> lmn; // angular momentum of x, y, z
        arma::vec alpha; // exponent
        arma::vec d_coe; // contraction coefficient
        int len; // number of primitive gaussians, the length of alpha, d_coe
        std::string lable; // lable of the AO, like s, px, py, pz, dxx, dxy, dxz, dyy, dyz, dzz, fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz...
        //constructor
        AO(arma::vec R0_input, arma::vec alpha_input, arma::vec d_input, arma::Col<unsigned int> lmn_input, const std::string & lable_input);
        ~AO(){}
        void printinfo();
};

//Sort the AOs according to lmn, s, px, py, pz, dxx, dxy, dxz, dyy, dyz, dzz, fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz...
int sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices, arma::uvec & sorted_offs);


// // a class to store the information of a pair of AO
// class AO_pair
// {
//     public:
//         const AO & ao1;
//         const AO & ao2;
//         //constructor
//         AO_pair(const AO & ao1_input, const AO & ao2_input): ao1(ao1_input), ao2(ao2_input){}
// };



#endif // AO_H