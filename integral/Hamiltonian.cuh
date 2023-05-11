#if !defined HAMILTONIAN_CUH
#define HAMILTONIAN_CUH


#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <basis/molecule_basis.cuh>
#include "Hamiltonian.h"


// To store the algorithm to evaluate the Hamiltonian matrix
class HamiltonianGPU: public Hamiltonian{
    public:
        Molecule_basisGPU m_molbasis_gpu; // the molecule basis on GPU

        HamiltonianGPU() = default;
        HamiltonianGPU(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7): Hamiltonian(m_molbasis_i, shreshold_i){};
        // We copy the molecule basis to GPU in the initialization function of child classes but we release it in the destructor here
        ~HamiltonianGPU() { release_molecule_basis_gpu(m_molbasis_gpu); }
};

class HartreeFock_Rys_gpu: public HamiltonianGPU{
    public:
        double *Schwarz_mat;  // the Schwarz matrix on GPU
        double *rys_root; // the rys root on GPU
        int Schwarz_mat_dim0, Schwarz_mat_dim1; // the dimension of Schwarz_mat , dim0 is outer dimension, dim1 is inner dimension (fast dimension)
        int rys_root_dim0, rys_root_dim1; // the dimension of rys_root , dim0 is outer dimension, dim1 is inner dimension (fast dimension)
        bool sort_AO; // whether to sort the AO in the molecule basis

        HartreeFock_Rys_gpu() = default;
        HartreeFock_Rys_gpu(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7, bool sort_AO_i = false): 
                    HamiltonianGPU(m_molbasis_i, shreshold_i), sort_AO(sort_AO_i){};
        // Release the memory of Schwarz_mat and rys_root on GPU in the destructor
        ~HartreeFock_Rys_gpu();
        // Allocate the memory of Schwarz_mat and rys_root on GPU in the initialization function
        virtual int init();
        // evaluate the Overlap matrix
        virtual int eval_OV(arma::mat &OV_mat);
        // evaluate the H core matrix (one-electron part)
        virtual int eval_Hcore(arma::mat &H_mat);
        // evaluate the G matrix (two-electron part)
        virtual int eval_G(arma::mat &P_mat, arma::mat &G_mat);
        virtual int eval_J(arma::mat &P_mat, arma::mat &J_mat);
        virtual int eval_K(arma::mat &P_mat, arma::mat &K_mat);
};


#endif // HAMILTONIAN_CUH