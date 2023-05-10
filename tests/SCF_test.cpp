#include <iomanip>
#include <filesystem>
#include <armadillo>
#include <Setup/Jobinfo.h>
#include <basis/molecule_basis.h>
#include <SCF/SCF.h>
#include <SCF/RSCF.h>
// #include <integral/JKmat.cpp>


using namespace std;


int run_SCF(std::string fin, std::string fout){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout))
      return 1;

    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    // MyMolBasis.PrintAll();

    // Read rem from file
    std::string algorithm = MyJob.GetRem("scf_algorithm");
    std::string method = MyJob.GetRem("method");
    std::cout << "scf_algorithm: " << algorithm << " method: " << method << "\n";

    std::cout <<std::setprecision(15);

    SCF* mySCF = new RSCF(MyMolBasis, 50, 1e-10, method, algorithm);
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
    // run_SCF("H2/H2_gpu.in", "H2_gpu.out")|
    // run_SCF("H2_6311g.in", "H2_6311g.out")|
    // run_SCF("H2_10.in", "H2_10.out")|
    // run_SCF("C2H4/C2H4.in", "C2H4.out")|
    run_SCF("C2H4/C2H4_gpu.in", "C2H4.out")|
    run_SCF("C4H10.in", "C4H10.out")|
    0;
}
