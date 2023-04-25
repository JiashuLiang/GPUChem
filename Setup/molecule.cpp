#include "molecule.h"
#include <map>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <stdio.h>



std::map<std::string, int> VAN_map { {"h", 1}, {"he", 2}, {"li", 3},  {"be", 4}, {"b", 5},   
                                    {"c", 6}, {"n", 7}, {"o", 8}, {"f", 9},  {"ne", 10}, 
                                    {"na", 11}, {"mg", 12}, {"al", 13}, {"si", 14},  {"p", 15}, 
                                    {"s", 16}, {"cl", 17}, {"ar", 18}, {"k", 19},  {"ca", 20}, 
                                    };

Atom::Atom(std::string atomname, arma::vec coord):m_coord(coord), m_ele_name(atomname){
    auto found_ele = VAN_map.find(atomname); 
    if(found_ele != VAN_map.end())
        m_effective_charge = found_ele->second;
    else{
        std::cerr << "Setup/molecule.cpp: Do not support atom " << atomname << "now.  The effective charge is set to 0\n ";
        m_effective_charge = 0;
    }
}

void Atom::PrintAtomInfo(){
  printf("%s, R( %1.4f, %1.4f, %1.4f), with effective charge: %d\n", m_ele_name.c_str(),
    m_coord(0), m_coord(1), m_coord(2), m_effective_charge);
}


void Molecule::reset(){
    mAtoms.clear();
    mAtoms.shrink_to_fit();
    m_charge = 0;
    num_nonpair_ele = 0;
    m_num_atoms = 0;
}

int Molecule::convert_from_strvecs(std::vector<std::string>::iterator mol_start, std::vector<std::string>::iterator mol_end){

  std::istringstream iss(*mol_start);
  if (!(iss >> m_charge >> num_nonpair_ele)){
      std::cerr << "Setup/molecule.cpp: There is some problem with molecule format.\n ";
      std::cerr << *mol_start;
      return 1;
  }
  num_nonpair_ele  -=1;


  for (std::vector<std::string>::iterator it = mol_start + 1; it !=mol_end; ++it)
  {
    std::istringstream iss(*it);
    arma::vec R0(3);
    // int AtomicN = 0;
    std::string atomname;
    if (!(iss >> atomname >> R0[0] >> R0[1] >> R0[2])){
          std::cerr << "Setup/molecule.cpp: There is some problem with AO format.\n ";
          std::cerr << *it;
          return 1;
      }
    
    Atom readAtom =Atom(atomname, R0);
    mAtoms.push_back(readAtom);
  }
    m_num_atoms = mAtoms.size();
    return 0;
}


void Molecule::PrintMoleculeInfo(){
  printf("Molecule have %d atoms, with charge %d , and %d single electron.\n", m_num_atoms, m_charge, num_nonpair_ele);

  for(auto atom: mAtoms)
    atom.PrintAtomInfo();
}

// int Molecule::read_from_file(std::string &fname){
//     reset();
//     double charge;
//     int nunum_nonpair_ele, num_Atoms;

// //   string line;
// //   getline(in, line);
// //   istringstream iss(line);

// }
