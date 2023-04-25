#ifndef UTIL_H
#define UTIL_H

#include <armadillo>
double Combination(int n, int k);
double DoubleFactorial(int n);

double Overlap_onedim(double xa, double xb, double alphaa, double alphab, int la, int lb);
double Overlap_3d(arma::vec &Ra, arma::vec &Rb, double alphaa, double alphab, arma::uvec &lmna, arma::uvec &lmnb);

// 2 electron integral of two primitive Gaussians (s-type orbitals)
double I2e_pG(arma::vec &Ra, arma::vec &Rb, double sigmaa, double sigmab);

#endif // UTIL_H