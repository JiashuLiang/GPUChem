#if !defined HAMILTONIAN_H
#define HAMILTONIAN_H


#include <armadillo>
#include <basis/molecule_basis.h>


// To store the algorithm to evaluate the Hamiltonian matrix
class Hamiltonian{
    public:
        Molecule_basis &m_molbasis; // molecule basis
        double shreshold; // shreshold to screen the small integrals

        Hamiltonian() = default;
        Hamiltonian(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7): m_molbasis(m_molbasis_i), shreshold(shreshold_i){};

        // initialize
        virtual int init()=0;
        // evaluate the Overlap matrix
        virtual int eval_OV(arma::mat &OV_mat)=0;
        // evaluate the H core matrix (one-electron part)
        virtual int eval_Hcore(arma::mat &H_mat)=0;
        // evaluate the G matrix (two-electron part)
        virtual int eval_G(arma::mat &P_mat, arma::mat &G_mat)=0;
        // evaluate the J matrix (Coulomb matrix)
        virtual int eval_J(arma::mat &P_mat, arma::mat &J_mat)=0;
        // evaluate the K matrix (Exchange matrix)
        virtual int eval_K(arma::mat &P_mat, arma::mat &K_mat)=0;
};


class HartreeFock_Rys: public Hamiltonian{
    public:
        arma::mat Schwarz_mat; // Schwarz matrix to screen the small integrals
        arma::mat rys_root; // Rys root to evaluate the two-electron integrals
        bool sort_AO; // whether to sort the AO

        HartreeFock_Rys() = default;
        HartreeFock_Rys(Molecule_basis &m_molbasis_i, double shreshold_i = 1e-7, bool sort_AO_i = false): 
                Hamiltonian(m_molbasis_i, shreshold_i), sort_AO(sort_AO_i){};
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


#endif // HAMILTONIAN_H