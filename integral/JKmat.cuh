#if !defined JKMAT_H
#define JKMAT_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <basis/molecule_basis.h>
#include <math.h>

int eval_Gmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &G_mat);
int eval_Jmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &J_mat);
int eval_Kmat_RSCF(Molecule_basis& system, arma::mat &rys_root, arma::mat& Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &K_mat);
int eval_Schwarzmat(Molecule_basis& system, arma::mat &rys_root, arma::mat &Schwarz_mat);

#endif // JKMAT_H