#if !defined RSCF_H
#define RSCF_H

#include <integral/Hamiltonian.h>
#include "SCF.h"

class RSCF: public SCF{
    public:
        SCF_algorithm *m_scf_algorithm;
        Hamiltonian *m_hamiltonian;
        Molecule_basis &m_molbasis;

        int dim, num_atoms, schwarz_tol;
        arma::mat S_mat, X_mat;
        arma::mat Ga;
        arma::mat H_core;
        arma::mat Pa, Pa_ascol;
        arma::mat Ca;
        arma::mat Fa;
        arma::vec Ea;
        double Ee, En, Etotal, E_one_ele, E_two_ele;


        RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence, 
            const std::string hamiltonian_name = "HF", const std::string scf_algorithm_name = "plain");
        ~RSCF();
        virtual int init();
        virtual int run();
        virtual double * getP_ptr();
        virtual double getEnergy();
        virtual int getdim(){return dim;}
        void UpdateEnergy();
        void UpdateFock();
        void UpdateDensity();
};


class RSCF_plain: public SCF_algorithm{
    public:
        RSCF *m_scf;

        int  max_iter;
        double tol, diff;

        virtual int init();
        virtual int run();
        RSCF_plain() = default;
        RSCF_plain(RSCF *m_scf_i, int max_it, double tolerence);
};




#endif // RSCF_H