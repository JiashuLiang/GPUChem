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
  printf("This AO info: %s, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x\n", lable.c_str(),
    R0(0), R0(1), R0(2), lmn(0), lmn(1), lmn(2));
  d_coe.t().print("d_coe");
  alpha.t().print("alpha");
}

AO::AO(arma::vec R0_input, arma::vec alpha_input,  arma::vec d_input, arma::Col<unsigned int> lmn_input, const std::string & lable_input):
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



int sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices, arma::uvec & sorted_offs){
//Sort the AOs according to lmn, s, px, py, pz, dxx, dxy, dxz, dyy, dyz, dzz, fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz...
// Now only support to sort s and p orbitals.
    // sorts AOs, s orbitals first then p orbitals next.
    // input: unsorted_AOs
    // output: sorted_AOs, sorted_indices, sorted_offs
    // return: 0 if success, 1 if error

    std::vector<AO> s_orbs, px_orbs, py_orbs, pz_orbs;
    std::vector<arma::uword> s_orbs_ind, px_orbs_ind, py_orbs_ind, pz_orbs_ind;
    //separate s, px, py, pz orbitals
    for (size_t mu = 0; mu < unsorted_AOs.size(); mu++){
        size_t lx = unsorted_AOs[mu].lmn(0);
        size_t ly = unsorted_AOs[mu].lmn(1);
        size_t lz = unsorted_AOs[mu].lmn(2);
        if (lx == 1 && ly == 0 && lz == 0){
            px_orbs.push_back(unsorted_AOs[mu]);
            px_orbs_ind.push_back(mu);
        } else if (lx == 0 && ly == 1 && lz == 0){
            py_orbs.push_back(unsorted_AOs[mu]);
            py_orbs_ind.push_back(mu);
        } else if (lx == 0 && ly == 0 && lz == 1){
            pz_orbs.push_back(unsorted_AOs[mu]);
            pz_orbs_ind.push_back(mu);
        } else if (lx == 0 && ly == 0 && lz == 0){
            s_orbs.push_back(unsorted_AOs[mu]);
            s_orbs_ind.push_back(mu);
        } else {
            throw std::runtime_error("basis/AO.cpp: Unsupported l_total in sort_AOs function.");
            return 1;
        }
    }
    assert(s_orbs.size() + px_orbs.size() + py_orbs.size() + pz_orbs.size() == unsorted_AOs.size());
    //combine s_orbs, px_orbs, py_orbs, pz_orbs into sorted_AOs
    sorted_AOs = s_orbs;
    sorted_AOs.insert(sorted_AOs.end(), px_orbs.begin(), px_orbs.end());
    sorted_AOs.insert(sorted_AOs.end(), py_orbs.begin(), py_orbs.end());
    sorted_AOs.insert(sorted_AOs.end(), pz_orbs.begin(), pz_orbs.end());
    
    //calculate sorted_offs
    sorted_offs.set_size(4);
    sorted_offs(0) = 0;
    sorted_offs(1) = s_orbs.size();
    sorted_offs(2) = s_orbs.size() + px_orbs.size();
    sorted_offs(3) = s_orbs.size() + px_orbs.size() + py_orbs.size();
    
    //combine s_orbs_ind, px_orbs_ind, py_orbs_ind, pz_orbs_ind into sorted_indices
    sorted_indices.set_size(s_orbs_ind.size() + px_orbs_ind.size() + py_orbs_ind.size() + pz_orbs_ind.size());
    sorted_indices.subvec(sorted_offs(0), sorted_offs(1) -1) = arma::conv_to<arma::uvec>::from(s_orbs_ind);
    if (px_orbs_ind.size() > 0)
        sorted_indices.subvec(sorted_offs(1), sorted_offs(2) -1) = arma::conv_to<arma::uvec>::from(px_orbs_ind);
    if (py_orbs_ind.size() > 0)
        sorted_indices.subvec(sorted_offs(2), sorted_offs(3) -1) = arma::conv_to<arma::uvec>::from(py_orbs_ind);
    if (pz_orbs_ind.size() > 0)
        sorted_indices.subvec(sorted_offs(3), sorted_indices.n_elem -1) = arma::conv_to<arma::uvec>::from(pz_orbs_ind);

    return 0;
}