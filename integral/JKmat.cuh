#if !defined JKMAT_H
#define JKMAT_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <basis/molecule_basis.cuh>
#include <math.h>

int eval_Gmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double *Pa_mat, double *G_mat, int rys_root_dim1);
int eval_Jmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
        double *Pa_mat, double *J_mat, int rys_root_dim1);
int eval_Kmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
        double *Pa_mat, double *K_mat, int rys_root_dim1);
__global__ void eval_Schwarzmat_GPU(AOGPU * mAOs, double *rys_root, double *Schwarz_mat, int nbasis, int rys_root_dim1);

#endif // JKMAT_H