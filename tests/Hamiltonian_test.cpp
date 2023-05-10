#include <iomanip>
#include <armadillo>
#include <Setup/Jobinfo.h>
#include <basis/molecule_basis.h>
#include <integral/Hamiltonian.h>
#include <integral/Hamiltonian.cuh>
#include <cmath>

using namespace std;

int run_Halmitonian(std::string test_case, std::string hamiltonian_name, double tol = 2e-6)
{

    std::string fin = test_case + "/" + test_case + ".in";
    std::string fout = test_case + "/" + test_case + ".out";
    std::string Pmat_name = test_case + "/Pmat.txt";
    std::string Energy_name = test_case + "/energy.txt";
    std::cout << "Input: " << fin << " Hamiltonian: " << hamiltonian_name << "\n";

    JobInfo MyJob;
    if (MyJob.read_from_file(fin, fout))
        return 1;
    Molecule_basis MyMolBasis(MyJob.m_molecule);
    MyMolBasis.Construct_basis(MyJob.GetRem("basis"));
    // MyMolBasis.PrintAll();

    std::cout << std::setprecision(15);

    Hamiltonian *myHamiltonian;
    if (hamiltonian_name == "hf")
        myHamiltonian = new HartreeFock_Rys(MyMolBasis, 1e-14, true);
    else if (hamiltonian_name == "hf_gpu")
        myHamiltonian = new HartreeFock_Rys_gpu(MyMolBasis, 1e-14, false);

    int ok = myHamiltonian->init();
    if (ok != 0){
        std::cout << "Hamiltonian init failed!" << std::endl;
        return 1;
    }

    size_t nbasis = MyMolBasis.get_basis_size();

    // Read Pmat from file
    arma::mat Pa(nbasis, nbasis);
    Pa.load(Pmat_name);
    // read ref energy from file
    arma::vec Energy_ref(4);
    Energy_ref.load(Energy_name);

    // arma::mat S_mat(nbasis,nbasis);
    // myHamiltonian->eval_OV(S_mat);
    // S_mat.print("Overlap matrix");
    arma::mat H_core(nbasis, nbasis), Ga(nbasis, nbasis);
    myHamiltonian->eval_Hcore(H_core);
    myHamiltonian->eval_G(Pa, Ga);
    arma::mat Ja(nbasis, nbasis), Ka(nbasis, nbasis);
    myHamiltonian->eval_J(Pa, Ja);
    myHamiltonian->eval_K(Pa, Ka);
    
    // Ja.print("Ja");

    // Check one electron energy
    double E_one_ele = arma::dot(Pa, H_core) * 2;
    if(std::abs(E_one_ele - Energy_ref(0)) > tol)
    {
        std::cout << "One electron energy is not correct!" << std::endl;
        std::cout << "Ref: " << Energy_ref(0) << " Calc: " << E_one_ele << std::endl;
        ok = 1;
    }
    // Check Coulomb and Exchange energy
    double Coulomb = arma::dot(Pa, Ja) * 2;
    double Exchange = -arma::dot(Pa, Ka);
    if(std::abs(Coulomb - Energy_ref(1)) > tol)
    {
        std::cout << "Coulomb energy is not correct!" << std::endl;
        std::cout << "Ref: " << Energy_ref(1) << " Calc: " << Coulomb << std::endl;
        ok = 1;
    }
    if(std::abs(Exchange - Energy_ref(2)) > tol)
    {
        std::cout << "Exchange energy is not correct!" << std::endl;
        std::cout << "Ref: " << Energy_ref(2) << " Calc: " << Exchange << std::endl;
        ok = 1;
    }
    // Check two electron energy
    double E_two_ele = arma::dot(Pa, Ga);
    if(std::abs(E_two_ele - Energy_ref(3)) > tol)
    {
        std::cout << "Two electron energy is not correct!" << std::endl;
        std::cout << "Ref: " << Energy_ref(3) << " Calc: " << E_two_ele << std::endl;
        std::cout << "Note: The calulation path of G_mat may be different from seperate calculations of J and K!" << std::endl;
        ok = 1;
    }
    
    if(ok == 0)
        std::cout << "All tests passed!" << std::endl;

    std::cout << std::endl;

    return ok;
}

int main(int argc, char *argv[])
{
    return
        // run_Halmitonian("H2", "hf")|
        // run_Halmitonian("H2", "hf_gpu")|
        run_Halmitonian("C2H4", "hf") |
        run_Halmitonian("C2H4", "hf_gpu")|
        0;
}
