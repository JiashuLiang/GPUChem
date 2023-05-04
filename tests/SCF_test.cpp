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

int run_SCF(std::string fin, std::string fout, std::string algorithm){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout))
      return 1;

    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    // MyMolBasis.PrintAll();

    std::cout << std::setprecision(15);

    // // print 2eints
    // size_t nbasis = MyMolBasis.get_basis_size();
    // // loading rys roots
    // std::string aux;
    // arma::mat rys_root;
    // if (const char* env_p = std::getenv("GPUChem_aux")){
    //     aux = std::string(env_p);
    //     if (!std::filesystem::is_directory(aux)) {
    //         throw std::runtime_error("basis/basis_set.cpp: The directory specified by GPUChem_aux does not exist!");
    //     }
    // }
    // rys_root.load(aux + "/rys_root.txt");

    // arma::mat mnsl(nbasis*nbasis, nbasis*nbasis);
    // for (size_t mu = 0; mu < nbasis; mu++)
    //     for (size_t nu = 0; nu < nbasis; nu++)
    //         for (size_t si = 0; si < nbasis; si++)
    //             for (size_t la = 0; la < nbasis; la++)
    //                 mnsl(mu*nbasis + nu, si*nbasis + la) = eval_2eint(rys_root, MyMolBasis.mAOs[mu], MyMolBasis.mAOs[nu], MyMolBasis.mAOs[si], MyMolBasis.mAOs[la]);
    // mnsl.raw_print("mnsl");

    SCF* mySCF = new RSCF(MyMolBasis, 50, 1e-9, "HF", algorithm);
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
    run_SCF("C2H4/C2H4.in", "C2H4.out")|
    // run_SCF("H2.in", "H2.out", "DIIS")|
    // run_SCF("H2_6311g.in", "H2_6311g.out", "DIIS")|
    // run_SCF("C2H4/C2H4.in", "C2H4.out", "DIIS")|
    // run_SCF("H2_10.in", "H2_10.out", "DIIS")|
    0;
}
