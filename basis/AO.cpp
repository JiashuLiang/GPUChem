#include "AO.h"
#include <stdexcept>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <cassert>
#include <string>
#include "util.h"
#include <vector>

using namespace std;

//AO functions
void AO::printinfo(){
  printf("This AO info: %s, R( %1.2f, %1.2f, %1.2f), with angular momentum: %lld %lld %lld\n", lable.c_str(),
    R0(0), R0(1), R0(2), lmn(0), lmn(1), lmn(2));
  d_coe.t().print("d_coe");
  alpha.t().print("alpha");
}

AO::AO(arma::vec R0_input, arma::vec alpha_input,  arma::vec d_input, arma::uvec lmn_input, const std::string & lable_input):
R0(R0_input), alpha(alpha_input), d_coe(d_input), lmn(lmn_input), lable(lable_input){
    assert(R0.n_elem == 3);
    assert(lmn.n_elem == 3);
    len = alpha.n_elem;
    assert(d_coe.n_elem == len);
    for (size_t k = 0; k <len; k++){
      double Overlap_Self = Overlap_3d(R0, R0, alpha(k), alpha(k), lmn, lmn);
      d_coe(k) /= sqrt(Overlap_Self);
    }
}

