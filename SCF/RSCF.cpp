#include "RSCF.h"
#include <armadillo>
#include <iostream>
#include <integral/Hamiltonian.h>

double hartree_to_ev = 27.211396641308;


RSCF::RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence,
            const std::string hamiltonian_name, const std::string scf_algorithm_name): m_molbasis(m_molbasis_i)
{
  nbasis = m_molbasis.mAOs.size();
  num_atoms = m_molbasis.m_mol.mAtoms.size();

  Pa = arma::zeros(nbasis, nbasis);
  Ga = arma::zeros(nbasis, nbasis);
  Ca.set_size(nbasis, nbasis);
  Ea.set_size(nbasis);

  H_core.set_size(nbasis, nbasis);
  S_mat.set_size(nbasis, nbasis);

  // std::cout << std::setprecision(3);
  // gamma.print("gamma");
  // S.print("Overlap");
  
  //Initialize the Hamiltonian
  if (hamiltonian_name == "HF")
    m_hamiltonian = new HartreeFock_Rys(m_molbasis, 1e-3 * tolerence);
  else
    m_hamiltonian = new HartreeFock_Rys(m_molbasis, 1e-3 * tolerence);


  //Initialize the SCF algorithm
  if (scf_algorithm_name == "DIIS")
    m_scf_algorithm = new RSCF_DIIS(this, max_it, tolerence, 4);
  else
    m_scf_algorithm = new RSCF_plain(this, max_it, tolerence);
    
  
  Ee = 0.;
  Etotal = 0.;

  // Nuclear Repulsion Energy
  En = 0.;
  for (size_t k = 0; k < num_atoms; k++){
    Atom & atom_k = m_molbasis.m_mol.mAtoms[k];
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
  //Initialize the Hamiltonian
  m_hamiltonian->init();

  //Evaluate the overlap matrix
  if (m_hamiltonian ->eval_OV(S_mat) != 0)
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

  if (m_hamiltonian ->eval_Hcore( H_core) != 0)
  {
    std::cerr << "Warn! H_core matrix evaluation is failed." << std::endl;
    return 1;
  }
  // H_core.print("H_core");

  return m_scf_algorithm->init();
}


int RSCF::run()
{
  return m_scf_algorithm->run();
}

void RSCF::UpdateEnergy(){
  E_two_ele = arma::dot(Pa, Ga);
  E_one_ele = arma::dot(Pa, H_core)* 2;
  Ee = E_two_ele + E_one_ele;
  Etotal = Ee + En;
}

void RSCF::UpdateFock(){
    m_hamiltonian->eval_G(Pa, Ga);
    Fa = H_core + Ga;
}

void RSCF::UpdateDensity(){
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

double * RSCF::getP_ptr(){
  return Pa.memptr();
}

