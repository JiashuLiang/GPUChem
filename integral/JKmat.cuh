#if !defined JKMAT_H
#define JKMAT_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <basis/molecule_basis.cuh>
#include <math.h>

// Evaluate the G matrix (two electron integrals) using the Rys quadrature method for the RSCF method
// schwarz_tol is the tolerance for the Schwarz screening, set integral to zero if it is smaller than the square of schwarz_tol
// schwarz_max is the maximum value of the Schwarz matrix
int eval_Gmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double *Pa_mat, double *G_mat, int rys_root_dim1);
// Evaluate the J matrix (Coulomb matrix) using the Rys quadrature method for the RSCF method
int eval_Jmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
        double *Pa_mat, double *J_mat, int rys_root_dim1);
// Evaluate the K matrix (Exchange matrix) using the Rys quadrature method for the RSCF method
int eval_Kmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
        double *Pa_mat, double *K_mat, int rys_root_dim1);

// Evaluate the Schwarz matrix for pre-screening the two-electron integrals
__global__ void eval_Schwarzmat_GPU(AOGPU * mAOs, double *rys_root, double *Schwarz_mat, int nbasis, int rys_root_dim1);

// I calculate JK together to save time, the functions above are JK separated
int eval_JKmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
		double *Pa_mat, double *J_mat, double *K_mat, int rys_root_dim1);
#endif // JKMAT_H