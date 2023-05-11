#include <stdlib.h>
#include <stdexcept>
#include <stdio.h>
#include <armadillo>
#include <vector>
#include <Setup/Jobinfo.h>
#include <basis/basis_set.h>
#include <basis/molecule_basis.h>

using namespace std;

// Check if the basis set is read correctly from the file
int Basis_reading(std::string basis_name){
    BasisSet MyBasis(basis_name);
    // MyBasis.PrintBasis();
    std::vector<std::string> SupportedElements = MyBasis.getSupportedElements();
    std::vector<BasisShell> C_basis = MyBasis.getElementBasis("c");
    std::vector<BasisShell> H_basis = MyBasis.getElementBasis("h");
    // print SupportedElements, C_basis, H_basis
    std::cout << "SupportedElements: ";
    for (auto it : SupportedElements){
      std::cout << it << " ";
    }
    std::cout << "\n";
    std::cout << "C_basis: ";
    for (auto it : C_basis){
      it.PrintShell();
    }
    std::cout << "\n";
    std::cout << "H_basis: ";
    for (auto it : H_basis){
      it.PrintShell();
    }

    return 0;
}

// Check if the basis set constructed for input file is correct
int Basis_reading(std::string fin, std::string fout){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout))
      return 1;

    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    MyMolBasis.PrintAll();

    return 0;
}



int main(int argc, char *argv[])
{
  return
    Basis_reading("sto3g")|
    Basis_reading("H2/H2.in", "H2.out")|
    Basis_reading("C2H4/C2H4.in", "C2H4.out")|
    0;
}
