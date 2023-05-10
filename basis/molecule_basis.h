#ifndef MOLECULE_BASIS_H
#define MOLECULE_BASIS_H

#include "AO.h"
#include "basis_set.h"
#include <Setup/molecule.h>
#include <armadillo>


// Molecule basis class
class Molecule_basis{
    public:
        std::vector<AO> mAOs; // All Atomic Orbitals
        Molecule m_mol; // Molecule Information

        // Sorted AOs
        std::vector<AO> mAOs_sorted;
        arma::uvec mAOs_sorted_index;  // Usually used for density matrix, P_mat = P_mat(mAOs_sorted_index, mAOs_sorted_index)
        arma::uvec mAOs_sorted_index_inv; // Inverse index of mAOs_sorted_index, usage: H_mat = H_mat(mAOs_sorted_index_inv, mAOs_sorted_index_inv)
        arma::uvec sorted_offs; // the offset of each kind of AOs in mAOs_sorted_index, for example, s orbital starts at sorted_offs(0), px orbital at sorted_offs(1), etc.
        
        int num_alpha_ele; // Number of alpha electrons
        int num_beta_ele;   // Number of beta electrons

        Molecule_basis() = default;
        Molecule_basis(std::vector<AO> AOs, Molecule mol, int a_ele, int b_ele): mAOs(AOs), num_alpha_ele(a_ele), num_beta_ele(b_ele), m_mol(mol){}
        Molecule_basis(Molecule mol);

        // Read basis file and Construct Molecule basis
        int Construct_basis(const std::string &basis_name);
        // Construct Molecule basis according to BasisSet input
        int Construct_basis(const BasisSet &basis_set);

        // Sort AOs and initialize mAOs_sorted, mAOs_sorted_index, mAOs_sorted_index_inv, sorted_offs
        int Sort_AOs();

        // Add a basis shell to mAOs
        int addBasisShell(const BasisShell & added_shell, const arma::vec &atom_position);

        void addAO(AO aAO){
            mAOs.push_back(aAO);
        }
        
        void PrintAll();
        void PrintBasisInfo();

        size_t get_basis_size(){
            return mAOs.size();
        }

        // Check whether to use restricted or unrestricted method
        bool HasNonPairElectron(){
            return (num_alpha_ele != num_beta_ele);
        }

};


#endif // MOLECULE_BASIS_H