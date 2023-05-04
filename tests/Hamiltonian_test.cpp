#include <iomanip>
#include <armadillo>
#include <Setup/Jobinfo.h>
#include <basis/molecule_basis.h>
#include <integral/Hamiltonian.h>


using namespace std;


int run_Halmitonian(std::string test_case){

    std::string fin = test_case + "/" + test_case + ".in";
    std::string fout = test_case + "/" + test_case + ".out";
    std::string Pmat_name = test_case + "/Pmat.txt";
    std::cout << "Input: " << fin << "\n";

    JobInfo MyJob;
    if(MyJob.read_from_file(fin, fout))
      return 1;
    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    // MyMolBasis.PrintAll();

    // Read Pmat from file
    arma::mat Pa;
    // Pa.load(Pmat_name);

    std::cout <<std::setprecision(15);

    Hamiltonian* myHamiltonian = new HartreeFock_Rys(MyMolBasis, 1e-14);
    int ok = myHamiltonian->init();
    if(ok != 0) return 1;
  
    size_t nbasis = MyMolBasis.get_basis_size();
    Pa.set_size(nbasis, nbasis);
    Pa.zeros();
    arma::mat H_core(nbasis,nbasis), Ga(nbasis,nbasis);
    myHamiltonian->eval_Hcore(H_core);
    myHamiltonian->eval_G(Pa, Ga);

    double E_one_ele = 0, E_two_ele = 0;
    E_two_ele = arma::dot(Pa, Ga);
    E_one_ele = arma::dot(Pa, H_core)* 2;
    std::cout << "One Electron Energy is " << E_one_ele << " hartree." << std::endl;
    std::cout << "Two electron Energy is " << E_two_ele << " hartree." << std::endl;

    arma::mat Ja(nbasis,nbasis), Ka(nbasis,nbasis);
    myHamiltonian->eval_J(Pa, Ja);
    myHamiltonian->eval_K(Pa, Ka);
    double Coulomb = arma::dot(Pa, Ja) * 2;
    double Exchange = - arma::dot(Pa, Ka);
    std::cout << "Coulomb Energy is " << Coulomb << " hartree." << std::endl;
    std::cout << "Exchange Energy is " << Exchange << " hartree." << std::endl;

    return 0;
}


int main(int argc, char *argv[])
{
  return
    run_Halmitonian("C2H4")|
    0;
}
