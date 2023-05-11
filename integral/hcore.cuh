#if !defined HCORE_CUH
#define HCORE_CUH
#define ARMA_ALLOW_FAKE_GCC

#include <basis/molecule_basis.cuh>
#include <armadillo>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace hcore_gpu {

// The function now in use

//main function
int eval_Hcoremat(Molecule_basisGPU& system, double * H_mat_gpu);
int eval_OVmat(Molecule_basisGPU& system, double * S_mat_gpu);
// the function that is called by the main function
__global__ void construct_T(double* Tmat, AOGPU* mAOs, size_t nbsf);
__global__ void construct_V(double* Vmat, AOGPU* mAOs, size_t nbsf, double* Atom_coords, const int* effective_charges, const int num_atom);
__global__ void construct_S_whole_mat(double* Smat,  AOGPU* mAOs, size_t nbsf);
// the device functions that are called by the construct_T and construct_V, and construct_S_whole_mat
__device__ double eval_Smunu(AOGPU &mu, AOGPU &nu);
__device__ double eval_Tmunu(AOGPU &mu, AOGPU &nu);
__device__ double eval_Vmunu(AOGPU &mu, AOGPU &nu, double* Atom_coords, const int* effective_charges, const int num_atom);

// math utils
__device__ int factorial (int n);

// S and T helper functions 
__device__ double poly_binom_expans_terms(int n, int ia, int ib, double PminusA_1d, double PminusB_1d);
__device__ double overlap_1d(int l1, int l2, double PminusA_1d, double PminusB_1d, double gamma);
__device__ double overlap(double* A,  int l1, int m1, int n1,double alpha, double* B, int l2, int m2, int n2,double beta );
__device__ double kinetic(double* A,int l1, int m1, int n1,double alpha, double* B, int l2, int m2, int n2, double beta);

// helpers for the incomplete gamma function
__device__ double gammln(double x);
__device__ void gser(double *gamser, double a, double x, double *gln);
__device__ void gcf(double *gammcf, double a, double x, double *gln);
__device__ double gammp(double a, double x);
__device__ double Fgamma(int m, double x);

// helpers for Nuclear attraction functions
__device__ double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma);
__device__ double nuclear_attraction(double *A,int l1, int m1, int n1,double alpha, double *B, int l2, int m2, int n2,double beta, double *C);




// additional GPU funcs to do calculations as ss, sp, ps, pp blocks
int eval_Hcoremat_sort_inside(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &H_mat);
int eval_OVmat_sort_inside(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &S_mat);

__device__ void construct_S_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid);
__global__ void construct_S(double* Smat,  AOGPU* mAOs, size_t nbsf, size_t p_start_ind);
__device__ void construct_T_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid);
__device__ void construct_V_block(double* Vmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, double* Atom_coords, const int* effective_charges, const int num_atom, size_t tid);
__global__ void construct_TV(double* Tmat, double* Vmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind, double* Atom_coords, const int* effective_charges, const int num_atom);

#endif // HCORE_CUH
}