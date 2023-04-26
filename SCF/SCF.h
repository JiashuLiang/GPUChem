#if !defined SCF_H
#define SCF_H


#include <armadillo>
#include <cassert>
#include <basis/molecule_basis.h>


class SCF{
    public:
        virtual int init()=0;
        virtual int run()=0;
        // return density matrix's pointer
        virtual double * getP_ptr()=0;
        virtual double getEnergy()=0;
        virtual int getdim()=0;
};

class RSCF: public SCF{
    private:
        Molecule_basis &m_molbasis;
        int dim, num_atoms, max_iter;
        double tol, diff;
        arma::mat S_mat, X_mat;
        arma::mat Ga;
        arma::mat H_core;
        arma::mat Pa, Pa_ascol;
        arma::mat Ca;
        arma::vec Ea;
        double Ee, En, Etotal, E_one_ele, E_two_ele;
    public:
        RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence);
        virtual int init();
        virtual int run();
        virtual double * getP_ptr();
        virtual double getEnergy();
        void UpdateEnergy();
        virtual int getdim(){return dim;}
};

// class USCF: public SCF{
//     private:
//         Molecule_basis &m_molbasis;
//         int dim, num_atoms, max_iter;
//         double tol;
//         arma::mat S_mat;
//         arma::mat H_core;
//         arma::mat Pa, Pb;
//         arma::mat Ga, Gb;
//         arma::mat Ca, Cb;
//         arma::vec Ea, Eb;
//         double Ee, En, Etotal;
//     public:
//         USCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence);
//         virtual int init();
//         virtual int run();
//         virtual arma::mat getPa() {return Pa;}
//         virtual arma::mat getPb() {return Pb;}
//         virtual double getEnergy();
// };






#endif // SCF_H