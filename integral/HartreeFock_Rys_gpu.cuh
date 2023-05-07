#if !defined HAMILTONIAN_CUH
#define HAMILTONIAN_CUH


#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <basis/molecule_basis.cuh>
#include "Hamiltonian.h"

// To store the algorithm to evaluate the Hamiltonian matrix
class HamiltonianGPU: public Hamiltonian{
    public:
        Molecule_basisGPU m_molbasis_gpu;

        HamiltonianGPU() = default;
        HamiltonianGPU(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7): Hamiltonian(m_molbasis_i, shreshold_i){
            copy_molecule_basis_to_gpu(m_molbasis, m_molbasis_gpu);
        };
        ~HamiltonianGPU() { release_molecule_basis_gpu(m_molbasis_gpu); }
};

class HartreeFock_Rys_gpu: public HamiltonianGPU{
    public:
        arma::mat Schwarz_mat;
        arma::mat rys_root;

        HartreeFock_Rys_gpu() = default;
        HartreeFock_Rys_gpu(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7): HamiltonianGPU(m_molbasis_i, shreshold_i){};
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