#if !defined JKMAT_H
#define JKMAT_H

#include <basis/molecule_basis.h>
#include <armadillo>
#include <math.h>

int eval_Gmat_RSCF(Molecule_basis& system, arma::mat &Pa_mat, arma::mat &G_mat);

#endif // JKMAT_H