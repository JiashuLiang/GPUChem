#if !defined RSCF_H
#define RSCF_H

#include <integral/Hamiltonian.h>
#include "SCF.h"

class RSCF: public SCF{
    public:
        SCF_algorithm *m_scf_algorithm; // the SCF algorithm
        Hamiltonian *m_hamiltonian;     // the Hamiltonian to get the integrals 
        Molecule_basis &m_molbasis;     // the molecule basis and geometry

        int nbasis, num_atoms, schwarz_tol; // number of basis functions, number of atoms, schwarz tolerance (used for integral screening)
        arma::mat S_mat, X_mat; // overlap matrix, overlap matrix's inverse square root
        arma::mat Ga; // the two electron integral matrix
        arma::mat H_core; // the one electron integral matrix
        arma::mat Pa; // density matrix
        arma::mat Ca; // the molecular orbital coefficient matrix
        arma::mat Fa; // the Fock matrix
        arma::vec Ea; // the energies of molecular orbitals
        double Ee, En, Etotal, E_one_ele, E_two_ele; // the electronic energy, nuclear repulsion energy, total energy, one electron energy, two electron energy


        RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence, 
            const std::string hamiltonian_name = "HF", const std::string scf_algorithm_name = "DIIS");
        ~RSCF();
        virtual int init();
        virtual int run();
        virtual double * getP_ptr();
        virtual double getEnergy();
        virtual int getnbasis(){return nbasis;}
        void UpdateEnergy();
        void UpdateFock();
        void UpdateDensity();
};

// The plain SCF algorithm for restricted SCF
class RSCF_plain: public SCF_algorithm{
    public:
        RSCF *m_scf; // the SCF object to get all needed data

        int  max_iter; // the maxium number of SCF iterations
        double tol, res_error; // energy difference tolerence and residual error
        // res_error is defined as arma::norm(m_scf->Pa- Pa_old, "fro") for now

        virtual int init();
        virtual int run();
        RSCF_plain() = default;
        RSCF_plain(RSCF *m_scf_i, int max_it, double tolerence);
};


// The DIIS SCF algorithm for restricted SCF
class RSCF_DIIS: public SCF_algorithm{
    public:
        RSCF *m_scf; // the SCF object to get all needed data

        int num_iter_each_DIIS; // the number of SCF iterations in one DIIS circle
        int  max_iter; // the maxium number of SCF iterations
        double tol, res_error; // energy difference tolerence and residual error
        // res_error is defined as arma::norm(Fa_r * m_scf->Pa * m_scf->S_mat - m_scf->S_mat * m_scf->Pa * Fa_r, "fro") for now

        virtual int init();
        virtual int run();
        RSCF_DIIS() = default;
        RSCF_DIIS(RSCF *m_scf_i, int max_it, double tolerence, int num_iter_each_DIIS_i = 5);

        // One DIIS iteration
        void DIIS(arma::mat &e, arma::vec &c);
};


#endif // RSCF_H