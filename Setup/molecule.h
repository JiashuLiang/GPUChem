#ifndef MOLECULE_H
#define MOLECULE_H

#include <armadillo>
#include <vector>

/*! Atom class to store Atom info */
class Atom{
    public:
        arma::vec m_coord; // atom Coordinates
        std::string m_ele_name; // Element name
        int m_effective_charge; // effective charges of atoms, usually the index of the atom in the periodic table, e.g. 1 for H, 6 for C, 8 for O, but could be other values due to the use of minimum basis
        Atom():m_ele_name("0"), m_effective_charge(0.0){}
        Atom(arma::vec coord): m_coord(coord), m_ele_name("0"), m_effective_charge(0.0){}
        Atom(std::string atomname, arma::vec coord, int VAN_i): m_coord(coord), m_ele_name(atomname), m_effective_charge(VAN_i){}
        Atom(std::string atomname, arma::vec coord);
        void PrintAtomInfo();
};


class Molecule{
    public:
        std::vector<Atom> mAtoms; // atoms in the molecule
        int m_charge; // total charge of the molecule
        int num_nonpair_ele; // number of single electrons in the molecule
        int m_num_atoms; // number of atoms in the molecule

        Molecule() = default;
        Molecule(std::vector<Atom> Atoms, int charge, int single_ele): mAtoms(Atoms), m_charge(charge), num_nonpair_ele(single_ele)
            {m_num_atoms = mAtoms.size();}
            
        void addAtom(Atom aAtom){
            mAtoms.push_back(aAtom);
        }
        // read the molecule from a vector of strings
        int convert_from_strvecs(std::vector<std::string>::iterator mol_start, std::vector<std::string>::iterator mol_end);
        // Clear the molecule info to re-read from file
        void reset();
        // Print the molecule info
        void PrintMoleculeInfo();

};

#endif // MOLECULE_H
