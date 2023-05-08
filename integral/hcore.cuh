#if !defined HCORE_CUH
#define HCORE_CUH
#define ARMA_ALLOW_FAKE_GCC

#include <basis/molecule_basis.cuh>
#include <armadillo>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// need vector class to make my code easier haha

namespace hcore_gpu {

class vector_gpu {
public:
    // Constructor
    __device__ vector_gpu(double* coords, int size) : m_coords(coords), m_size(size) {}

    // Accessors
    __device__ double* coords() const { return m_coords; }
    __device__ int size() const { return m_size; }

    // vector_gpu addition and subtraction operators
    __device__ vector_gpu operator+(const vector_gpu& other) const {
        double* result_coords = new double[m_size];
        for (int i = 0; i < m_size; ++i) {
            result_coords[i] = m_coords[i] + other.m_coords[i];
        }
        return vector_gpu(result_coords, m_size);
    }

    __device__ vector_gpu operator-(const vector_gpu& other) const {
        double* result_coords = new double[m_size];
        for (int i = 0; i < m_size; ++i) {
            result_coords[i] = m_coords[i] - other.m_coords[i];
        }
        return vector_gpu(result_coords, m_size);
    }

    // vector_gpu norm calculation
    __device__ double norm() const {
        double sum_of_squares = 0.0;
        for (int i = 0; i < m_size; ++i) {
            sum_of_squares += m_coords[i] * m_coords[i];
        }
        return sqrtf(sum_of_squares);
    }
    // Scalar multiplication operator
    __device__ vector_gpu operator*(double scalar) const {
        double* result_coords = new double[m_size];
        for (int i = 0; i < m_size; ++i) {
            result_coords[i] = m_coords[i] * scalar;
        }
        return vector_gpu(result_coords, m_size);
    }
    // Destructor
    __device__ ~vector_gpu() {
        delete[] m_coords;
    }

private:
    double* m_coords;
    int m_size;
};

// Primary functions
int eval_Hcoremat(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &H_mat);
// int eval_Hcoremat(Molecule_basis& system, arma::mat &H_mat);

int eval_OVmat(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &S_mat);

// size_t sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices);
size_t sort_AOs(AOGPU* unsorted_AOs, const int nbsf, std::vector<AOGPU> &sorted_AOs, arma::uvec &sorted_indices);
size_t sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices);

// void construct_S(arma::mat &Smat, std::vector<AO> &mAOs, size_t p_start_ind);
// void construct_V(arma::mat &Vmat, std::vector<AO> &mAOs, size_t p_start_ind, const Molecule &mol);
// void construct_T(arma::mat &Tmat, std::vector<AO> &mAOs, size_t p_start_ind);
// void construct_S_unsorted(arma::mat &Smat, std::vector<AO> &mAOs);
// void construct_V_unsorted(arma::mat &Vmat, std::vector<AO> &mAOs, const Molecule &mol);
// void construct_T_unsorted(arma::mat &Tmat, std::vector<AO> &mAOs);


__device__ double eval_Smunu(AOGPU &mu, AOGPU &nu);
__device__ double eval_Tmunu(AOGPU &mu, AOGPU &nu);
__device__ double eval_Vmunu(AOGPU &mu, AOGPU &nu, double* Atom_coords, const int* effective_charges, const int num_atom);

// math utils
__device__ int factorial (int n);
__device__ int nCk (int n, int k);

// S and T helper functions 
__device__ vector_gpu gaussian_product_center(double alpha, vector_gpu &A, double beta, vector_gpu &B);
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

// Nuclear attraction functions
__device__ double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma);
__device__ vector_gpu A_tensor(int l1, int l2, double PA, double PB, double CP, double g);
__device__ double nuclear_attraction(double *A,int l1, int m1, int n1,double alpha, double *B, int l2, int m2, int n2,double beta, double *C);


// additional GPU funcs

__device__ void construct_S_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid);
__global__ void construct_S(double* Smat,  AOGPU* mAOs, size_t nbsf, size_t p_start_ind);
__device__ void construct_T_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid);
__device__ void construct_V_block(double* Vmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, double* Atom_coords, const int* effective_charges, const int num_atom, size_t tid);
__global__ void construct_TV(double* Tmat, double* Vmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind, double* Atom_coords, const int* effective_charges, const int num_atom);

__global__ void construct_T(double* Tmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind);
__global__ void construct_V(double* Vmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind, double* Atom_coords, const int* effective_charges, const int num_atom);

#endif // HCORE_CUH
}