#include "molecule_basis.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>


Molecule_basis::Molecule_basis(Molecule mol): m_mol(mol) {
    int num_ele = 0;
    for (const auto& atom : m_mol.mAtoms) {
        num_ele += atom.m_effective_charge;
    }
    num_ele -= m_mol.m_charge;
    // check whether num_ele + m_mol.num_nonpair_ele is even
    if ((num_ele + m_mol.num_nonpair_ele) % 2 != 0) {
        throw std::runtime_error("basis/basis_set.cpp: The number of non-pair electrons is not correct!");
    }
    // std::cout << "num_ele: " << num_ele << " num_nonpair_ele: " << m_mol.num_nonpair_ele << std::endl;
    num_alpha_ele = (num_ele + m_mol.num_nonpair_ele) / 2;
    num_beta_ele = (num_ele - m_mol.num_nonpair_ele) / 2;

}

int Molecule_basis::addBasisShell(const BasisShell & added_shell, const arma::vec &atom_position){
    std::string shelltype = added_shell.Shell_type;
    // Choosing different function according to shelltype
    if (shelltype == "s") {
        // s-type
        arma::Col<unsigned int> lmn={0,0,0};
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), lmn, "s"));
    } else if (shelltype == "p") {
        // p-type
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,0,0}), "px"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,1,0}), "py"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,0,1}), "pz"));
    } else if (shelltype == "sp") {
        // sp-type
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,0,0}), "s"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(1), arma::Col<unsigned int>({1,0,0}), "px"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(1), arma::Col<unsigned int>({0,1,0}), "py"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(1), arma::Col<unsigned int>({0,0,1}), "pz"));
    } else if (shelltype == "d") {
        // d-type
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({2,0,0}), "dxx"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,1,0}), "dxy"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,0,1}), "dxz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,2,0}), "dyy"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,1,1}), "dyz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,0,2}), "dzz"));
    } else if (shelltype == "f") {
        // f-type
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({3,0,0}), "fxxx"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({2,1,0}), "fxxy"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({2,0,1}), "fxxz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,2,0}), "fxyy"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,1,1}), "fxyz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({1,0,2}), "fxzz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,3,0}), "fyyy"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,2,1}), "fyyz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,1,2}), "fyyz"));
        mAOs.push_back(AO(atom_position, added_shell.exponents, added_shell.coefficients.col(0), arma::Col<unsigned int>({0,0,3}), "fzzz"));
    } else {
        std::cerr << "Error: Shell type " << shelltype << " is not supported." << std::endl;
        return 1;
    }
    return 0;
}


int Molecule_basis::Construct_basis(const std::string& basis_name) {
    BasisSet basis_set(basis_name);
    return Construct_basis(basis_set);
}

int Molecule_basis::Construct_basis(const BasisSet& basis_set) {
    // Iterate through each atom in the molecule
    for (const auto& atom : m_mol.mAtoms) {
        const std::string& element = atom.m_ele_name;

        // Check if the element is supported by the basis set
        if (!basis_set.isElementSupported(element)) {
            std::cerr << "Error: Element " << element << " is not supported by the basis set." << std::endl;
            return 1;
        }

        // Retrieve the basis functions for the element
        std::vector<BasisShell> element_basis = basis_set.getElementBasis(element);

        // Iterate through shell of the element
        for (const auto& shell : element_basis) {
            // Create new AO objects as the basis functions
            addBasisShell(shell, atom.m_coord);
        }
    }
    return 0;
}

void Molecule_basis::PrintBasisInfo(){
    std::cout << "Number of basis functions: " << mAOs.size() << std::endl;
    for(int i=0; i<mAOs.size(); i++){
        std::cout << "AO " << i << " : " << std::endl;
        mAOs[i].printinfo();    
    }
}

void Molecule_basis::PrintAll(){
    // Print the molecule information
    std::cout << "Molecule information:" << std::endl;
    m_mol.PrintMoleculeInfo();
    // Print basis information
    std::cout << "Number of alpha electrons: " << num_alpha_ele << std::endl;
    std::cout << "Number of beta electrons: " << num_beta_ele << std::endl;
    PrintBasisInfo();
}

int Molecule_basis::Sort_AOs(){
    int ok = sort_AOs(mAOs, mAOs_sorted, mAOs_sorted_index, sorted_offs);
    mAOs_sorted_index_inv = arma::sort_index(mAOs_sorted_index);
    return ok;
}
