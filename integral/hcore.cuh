#if !defined HCORE_CUH
#define HCORE_CUH

#include <basis/molecule_basis.h>
#include <armadillo>
#include <math.h>

// Primary functions
int eval_Hcoremat(Molecule_basis& system, arma::mat &H_mat);
int eval_OVmat(Molecule_basis& system, arma::mat &S_mat);

size_t sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices);
void construct_S(arma::mat &Smat, std::vector<AO> &mAOs, size_t p_start_ind);
void construct_V(arma::mat &Vmat, std::vector<AO> &mAOs, size_t p_start_ind, const Molecule &mol);
void construct_T(arma::mat &Tmat, std::vector<AO> &mAOs, size_t p_start_ind);
void construct_S_unsorted(arma::mat &Smat, std::vector<AO> &mAOs);
void construct_V_unsorted(arma::mat &Vmat, std::vector<AO> &mAOs, const Molecule &mol);
void construct_T_unsorted(arma::mat &Tmat, std::vector<AO> &mAOs);


double eval_Smunu(AO &mu, AO &nu);
double eval_Tmunu(AO &mu, AO &nu);
double eval_Vmunu(AO &mu, AO &nu, const Molecule &mol);

// math utils
int factorial (int n);
int nCk (int n, int k);

// S and T helper functions 
__device__ arma::vec gaussian_product_center(double alpha, arma::vec &A, double beta, arma::vec &B);
__device__ double poly_binom_expans_terms(int n, int ia, int ib, double PminusA_1d, double PminusB_1d);
__device__ double overlap_1d(int l1, int l2, double PminusA_1d, double PminusB_1d, double gamma);
__device__ double overlap(arma::vec A,  int l1, int m1, int n1,double alpha, arma::vec B, int l2, int m2, int n2,double beta );
__device__ double kinetic(arma::vec A,int l1, int m1, int n1,double alpha, arma::vec B, int l2, int m2, int n2,double beta );



// helpers for the incomplete gamma function
__device__ double gammln(double x);
__device__ void gser(double *gamser, double a, double x, double *gln);
__device__ void gcf(double *gammcf, double a, double x, double *gln);
__device__ double gammp(double a, double x);
__device__ double Fgamma(int m, double x);

// Nuclear attraction functions
__device__ double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma);
__device__ arma::vec A_tensor(int l1, int l2, double PA, double PB, double CP, double g);
__device__ double nuclear_attraction(arma::vec &A,int l1, int m1, int n1,double alpha, arma::vec &B, int l2, int m2, int n2,double beta, arma::vec &C);


// additional GPU funcs


__device__ void construct_T_block(double* Tmat,  AO* mAOs, size_t mu_start_ind, size_t nu_start_ind size_t num_mu, size_t num_nu);
__device__ void construct_V_block(double* Vmat,  AO* mAOs, size_t mu_start_ind, size_t nu_start_ind size_t num_mu, size_t num_nu, const Molecule &mol);
__global__ void construct_TV(double* T_mat_gpu, double* V_mat_gpu, AO* mAOs, size_t nbsf, size_t p_start_ind, const Molecule mol);
#endif // HCORE_CUH