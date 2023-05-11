#if !defined JKMAT_H
#define JKMAT_H

#include <basis/molecule_basis.h>
#include <armadillo>
#include <math.h>

// Evaluate the G matrix (two electron integrals) using the Rys quadrature method for the RSCF method
// schwarz_tol is the tolerance for the Schwarz screening, set integral to zero if it is smaller than the square of schwarz_tol
// schwarz_max is the maximum value of the Schwarz matrix
int eval_Gmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &G_mat);
// Evaluate the J matrix (Coulomb matrix) using the Rys quadrature method for the RSCF method
int eval_Jmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &J_mat);
// Evaluate the K matrix (Exchange matrix) using the Rys quadrature method for the RSCF method
int eval_Kmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &K_mat);

// Evaluate the Schwarz matrix for pre-screening the two-electron integrals
int eval_Schwarzmat(Molecule_basis& system, arma::mat &rys_root, arma::mat &Schwarz_mat);

// Evaluate the G matrix Using mAOs directly instead of Molecule_basis
// I calculate JK together to save time, the functions above are JK separated
int eval_Gmat_RSCF(std::vector<AO> &mAOs, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &G_mat);
// Evaluate the JK matrix Using mAOs directly instead of Molecule_basis
int eval_JKmat_RSCF(std::vector<AO> &mAOs, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol,
     arma::mat &Pa_mat, arma::mat &J_mat, arma::mat &K_mat);

// calculates (i j | k l), each of those is a contracted GTO basis function
// sorry using i j k l here instead of mu nu si la
double eval_2eint(arma::mat &rys_root, AO &AO_i, AO &AO_j, AO &AO_k, AO &AO_l);

#endif // JKMAT_H