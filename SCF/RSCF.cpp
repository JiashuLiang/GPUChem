#include "RSCF.h"
#include <armadillo>
#include <iostream>
#include <chrono>
#include <integral/Hamiltonian.h>
#include <integral/Hamiltonian.cuh>

double hartree_to_ev = 27.211396641308;

RSCF::RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence,
           const std::string hamiltonian_name, const std::string scf_algorithm_name) : m_molbasis(m_molbasis_i)
{
    nbasis = m_molbasis.mAOs.size();
    num_atoms = m_molbasis.m_mol.mAtoms.size();

    // initialize the matrices
    Pa = arma::zeros(nbasis, nbasis);
    Ga = arma::zeros(nbasis, nbasis);
    Ca.set_size(nbasis, nbasis);
    Ea.set_size(nbasis);
    H_core.set_size(nbasis, nbasis);
    S_mat.set_size(nbasis, nbasis);

    // std::cout << std::setprecision(3);

    // Create the Hamiltonian object
    if (hamiltonian_name == "hf_gpu")
        m_hamiltonian = new HartreeFock_Rys_gpu(m_molbasis, 1e-3 * tolerence, true);
    else
        m_hamiltonian = new HartreeFock_Rys(m_molbasis, 1e-3 * tolerence, true);

    // Create the SCF algorithm object
    if (scf_algorithm_name == "diis")
        m_scf_algorithm = new RSCF_DIIS(this, max_it, tolerence, 4);
    else
        m_scf_algorithm = new RSCF_plain(this, max_it, tolerence);

    Ee = 0.;
    Etotal = 0.;

    // Nuclear Repulsion Energy
    En = 0.;
    for (size_t k = 0; k < num_atoms; k++)
    {
        Atom &atom_k = m_molbasis.m_mol.mAtoms[k];
        for (size_t j = 0; j < k; j++)
        {
            arma::vec Ra = atom_k.m_coord, Rb = m_molbasis.m_mol.mAtoms[j].m_coord;
            double Rd = arma::norm(Ra - Rb, 2);
            En += atom_k.m_effective_charge * m_molbasis.m_mol.mAtoms[j].m_effective_charge / Rd;
        }
    }
}

int RSCF::init()
{
    auto start_time_t = std::chrono::steady_clock::now();

    // Initialize the Hamiltonian
    m_hamiltonian->init();

    // Evaluate the overlap matrix
    if (m_hamiltonian->eval_OV(S_mat) != 0)
    {
        std::cerr << "Warn! Overlap matrix evaluation is failed." << std::endl;
        return 1;
    }
    // S_mat.print("S_mat");

    // Calculate X_mat = S_mat^(-1/2)
    arma::vec S_eigval;
    // Use H_core as a temporary matrix to store eigenvectors
    arma::eig_sym(S_eigval, H_core, S_mat);
    X_mat = H_core * arma::diagmat(arma::pow(S_eigval, -0.5)) * H_core.t();

    if (m_hamiltonian->eval_Hcore(H_core) != 0)
    {
        std::cerr << "Warn! H_core matrix evaluation is failed." << std::endl;
        return 1;
    }
    // H_core.print("H_core");

    // Initialize the SCF algorithm object
    int ok = m_scf_algorithm->init();

    auto end_time_t = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time_t - start_time_t;
    std::cout << "SCF initialization is finished in " << elapsed_seconds.count() << " seconds." << std::endl;

    return ok;
}

int RSCF::run()
{
    auto start_time_t = std::chrono::steady_clock::now();
    int ok = m_scf_algorithm->run();
    auto end_time_t = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time_t - start_time_t;
    std::cout << "SCF calculation is finished in " << elapsed_seconds.count() << " seconds." << std::endl;

    return ok;
}

void RSCF::UpdateEnergy()
{
    // Ee = 2 * sum_{ij}^{occ} Pa_{ij} * (H_{ij} + F_{ij}) = 2 * sum_{ij}^{occ} Pa_{ij} * H_{ij} + sum_{ij}^{occ} Pa_{ij} * G_{ij}
    E_two_ele = arma::dot(Pa, Ga);
    E_one_ele = arma::dot(Pa, H_core) * 2;
    Ee = E_two_ele + E_one_ele;
    Etotal = Ee + En;
}

void RSCF::UpdateFock()
{
    m_hamiltonian->eval_G(Pa, Ga);
    Fa = H_core + Ga;
}

void RSCF::UpdateDensity()
{
    Pa = Ca.cols(0, m_molbasis.num_alpha_ele - 1) * Ca.cols(0, m_molbasis.num_alpha_ele - 1).t();
}

double RSCF::getEnergy()
{
    std::cout << "Nuclear Repulsion Energy is " << En << " hartree." << std::endl;
    std::cout << "One Electron Energy is " << E_one_ele << " hartree." << std::endl;
    std::cout << "Two electron Energy is " << E_two_ele << " hartree." << std::endl;
    std::cout << "Total Electron Energy is " << Ee << " hartree." << std::endl;
    return Etotal;
}

double *RSCF::getP_ptr()
{
    return Pa.memptr();
}
