#ifndef MOLECULE_BASIS_H
#define MOLECULE_BASIS_H

#include "AO.h"
#include "basis_set.h"
#include <Setup/molecule.h>
#include <armadillo>

class Molecule_basis{
    public:
        std::vector<AO> mAOs;
        Molecule m_mol;
        int num_alpha_ele;
        int num_beta_ele;

        Molecule_basis() = default;
        Molecule_basis(std::vector<AO> AOs, Molecule mol, int a_ele, int b_ele): mAOs(AOs), num_alpha_ele(a_ele), num_beta_ele(b_ele), m_mol(mol){}
        Molecule_basis(Molecule mol);

        // Read basis file and Construct Molecule basis
        int Construct_basis(const std::string &basis_name);
        // Construct Molecule basis according to BasisSet input
        int Construct_basis(const BasisSet &basis_set);

        int addBasisShell(const BasisShell & added_shell, const arma::vec &atom_position);
        void addAO(AO aAO){
            mAOs.push_back(aAO);
        }
        
        void PrintAll();
        void PrintBasisInfo();
        // For get something as matrix for better performance
        // void get_something(arma::mat &Something);

        size_t get_basis_size(){
            return mAOs.size();
        }

        bool HasNonPairElectron(){
            return (num_alpha_ele != num_beta_ele);
        }

};


#endif // MOLECULE_BASIS_H