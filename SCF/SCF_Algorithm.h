#if !defined SCF_ALGORITHM_H
#define SCF_ALGORITHM_H


#include <armadillo>
#include <cassert>
#include <basis/molecule_basis.h>


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






#endif // SCF_ALGORITHM_H