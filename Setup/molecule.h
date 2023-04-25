#ifndef MOLECULE_H
#define MOLECULE_H

#include <armadillo>
#include <vector>

/*! Atom class to store Atom info */
class Atom{
    public:
        arma::vec m_coord; // Coordinates
        std::string m_ele_name; // Element name
        int m_effective_charge; // Valence atomic number
        Atom():m_ele_name("0"), m_effective_charge(0.0){}
        Atom(arma::vec coord): m_coord(coord), m_ele_name("0"), m_effective_charge(0.0){}
        Atom(std::string atomname, arma::vec coord, int VAN_i): m_coord(coord), m_ele_name(atomname), m_effective_charge(VAN_i){}
        Atom(std::string atomname, arma::vec coord);
        void PrintAtomInfo();
};


class Molecule{
    public:
        std::vector<Atom> mAtoms;
        int m_charge;
        int num_nonpair_ele;
        int m_num_atoms;

        Molecule() = default;
        Molecule(std::vector<Atom> Atoms, int charge, int single_ele): mAtoms(Atoms), m_charge(charge), num_nonpair_ele(single_ele)
            {m_num_atoms = mAtoms.size();}
            
        void addAtom(Atom aAtom){
            mAtoms.push_back(aAtom);
        }
        // Molecule(std::string &fname); //Read from file
        // int read_from_file(std::string &fname);
        int convert_from_strvecs(std::vector<std::string>::iterator mol_start, std::vector<std::string>::iterator mol_end);
        void reset();
        void PrintMoleculeInfo();

};

#endif // MOLECULE_H
