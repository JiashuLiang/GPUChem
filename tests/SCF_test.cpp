#include <stdlib.h>
#include <stdexcept>
#include <stdio.h>
#include <armadillo>
#include <vector>
#include <Setup/Jobinfo.h>
#include <basis/molecule_basis.h>
#include <SCF/SCF.h>
#include <SCF/RSCF.h>


using namespace std;


int run_SCF(std::string fin, std::string fout){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout))
      return 1;

    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    // MyMolBasis.PrintAll();

    SCF* mySCF = new RSCF(MyMolBasis, 50, 1e-5, "HF", "plain");
    int ok = mySCF->init();
    if(ok != 0) return 1;
    ok = mySCF->run();
    if(ok != 0) return 1;
    double Energy = mySCF->getEnergy();
    std::cout<< "The molecule has energy "<< Energy << std::endl;


    return 0;
}



int main(int argc, char *argv[])
{
  return
    // run_SCF("H2_6311g.in", "H2_6311g.out")|
    // run_SCF("H2_10.in", "H2_10.out")|
    run_SCF("C2H4.in", "C2H4.out")|
    0;
}
