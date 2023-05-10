#if !defined SCF_H
#define SCF_H


#include <armadillo>
#include <cassert>
#include <basis/molecule_basis.h>

// The SCF class is an abstract class for all SCFs, like USCF, ROHF, etc.
// its main purpose is to provide a common interface for all SCF
// Its child class may contain the objects needed for SCF, like the density matrix, Fock matrix, etc.
class SCF{
    public:
        virtual int init()=0; // initialize the SCF algorithm
        virtual int run()=0; // run the SCF algorithm
        
        virtual double * getP_ptr()=0; // return density matrix's pointer
        virtual double getEnergy()=0;
        virtual int getnbasis()=0;
};



// The SCF_algorithm class is an abstract class for all SCF algorithms, like plain SCF, DIIS, etc.
// its main purpose is to provide a common interface for all SCF algorithms
class SCF_algorithm{
    public:
        virtual int init()=0;
        virtual int run()=0;
};


// This project only implements the RSCF 
// class USCF: public SCF{
//     private:
//         Molecule_basis &m_molbasis;
//         int nbasis, num_atoms, max_iter;
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