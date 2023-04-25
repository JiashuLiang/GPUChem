#include "SCF.h"
#include <stdio.h>
#include <math.h>
#include <cassert>
#include <armadillo>
#include <iostream>
#include <vector>
#include <integral/hcore.h>
#include <integral/JKmat.h>

double hartree_to_ev = 27.211396641308;


RSCF::RSCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence) : m_molbasis(m_molbasis_i), max_iter(max_it), tol(tolerence)
{
  dim = m_molbasis.mAOs.size();
  num_atoms = m_molbasis.m_mol.mAtoms.size();

  Pa = arma::zeros(dim, dim);
  Ga = arma::zeros(dim, dim);
  Ca.set_size(dim, dim);
  Ea.set_size(dim);

  H_core.set_size(dim, dim);
  S_mat.set_size(dim, dim);

  // std::cout << std::setprecision(3);
  // gamma.print("gamma");
  // S.print("Overlap");
  
  
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
  //Do H_core part
  size_t k_AO = 0;
  if (eval_OVmat(m_molbasis, S_mat) != 0)
  {
    std::cerr << "Warn! Overlap matrix evaluation is failed." << std::endl;
    return 1;
  }
  S_mat.print("S_mat");
  // Calculate X_mat = S_mat^(-1/2)
  arma::vec S_eigval;
  // Use H_core as a temporary matrix to store eigenvectors
  arma::eig_sym(S_eigval, H_core, S_mat);
  X_mat = H_core * arma::diagmat(arma::pow(S_eigval, -0.5)) * H_core.t();
  if (eval_Hcoremat(m_molbasis, H_core) != 0)
  {
    std::cerr << "Warn! H_core matrix evaluation is failed." << std::endl;
    return 1;
  }
  H_core.print("H_core");

  // Initial guess for Pa use Ca = I
  Ca.eye();
  Pa = Ca.cols(0, m_molbasis.num_alpha_ele - 1) * Ca.cols(0, m_molbasis.num_alpha_ele - 1).t();

  return 0;
}


int RSCF::run()
{
  // std::cout << "SCF iteration starts." << std::endl;
  // Get initial guess for Ga
  eval_Gmat_RSCF(m_molbasis, Pa, Ga);
  arma::mat Fa = H_core + Ga;
  arma::mat Pa_old;
  size_t k = 0;
  for (; k < max_iter; k++)
  {
    std::cout << "Iteration: " << k << std::endl;
    Pa.print("Pa");
    Ga.print("Ga");
    Fa.print("Fa");
    Pa_old = Pa;
    Fa = X_mat.t() * Fa * X_mat; // Get Fa' = X_mat^(-1) * Fa * X_mat^(-1)
    Fa.print("Fa'");
    arma::eig_sym(Ea, Ca, Fa);  // Solve eigen equation Fa' * Ca' = Ea * Ca'
    // Ea.print("Ea");
    Ca = X_mat * Ca; // Get Ca = X_mat * Ca'
    Pa = Ca.cols(0, m_molbasis.num_alpha_ele - 1) * Ca.cols(0, m_molbasis.num_alpha_ele - 1).t();
    // std::cout << "after solving eigen equation: " << k << std::endl;
    // Ca.print("Ca");
    if (arma::approx_equal(Pa, Pa_old, "absdiff", tol))
      break;
    // Pa.print("Pa_new");
    eval_Gmat_RSCF(m_molbasis, Pa, Ga); // Get new Ga
    Fa = H_core + Ga;
  }
  if (k == max_iter)
  {
    std::cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
    return 1;
  }
  Ea.print("Ea");
  Ca.print("Ca");
  return 0;
}

double RSCF::getEnergy()
{
  Ee = arma::dot(Pa, Ga);
  Ee += arma::dot(Pa, H_core)* 2;
  Etotal = Ee + En;
  std::cout << "Nuclear Repulsion Energy is " << En << " hartree." << std::endl;
  std::cout << "Electron Energy is " << Ee << " hartree." << std::endl;
  return Etotal;
}

double * RSCF::getP_ptr(){
  return Pa.memptr();
}


// USCF::USCF(Molecule_basis &m_molbasis_i, int max_it, double tolerence) : m_molbasis(m_molbasis_i), max_iter(max_it), tol(tolerence)
// {
//   dim = m_molbasis.mAOs.size();
//   num_atoms = m_molbasis.m_mol.mAtoms.size();

//   Pa = arma::zeros(dim, dim);
//   Pb = arma::zeros(dim, dim);
//   Ga = arma::zeros(dim, dim);
//   Gb = arma::zeros(dim, dim);
//   Ca.set_size(dim, dim);
//   Cb.set_size(dim, dim);
//   Ea.set_size(dim);
//   Eb.set_size(dim);

//   H_core.set_size(dim, dim);
//   S_mat.set_size(dim, dim);
//   G_mat.set_size(dim, dim);

//   // std::cout << std::setprecision(3);
//   // gamma.print("gamma");
//   // S.print("Overlap");
  
  
//   Ee = 0.;
//   Etotal = 0.;

//   // Nuclear Repulsion Energy
//   En = 0.;
//   for (size_t k = 0; k < num_atoms; k++){
//     Atom & atom_k = m_molbasis.m_mol.mAtoms[k];
//     for (size_t j = 0; j < k; j++)
//     {
//       arma::vec Ra = atom_k.m_coord, Rb = m_molbasis.m_mol.mAtoms[j].m_coord;
//       double Rd = arma::norm(Ra - Rb, 2);
//       En += atom_k.m_effective_charge * m_molbasis.m_mol.mAtoms[j].m_effective_charge / Rd;
//     }
//   }
// }

// int USCF::init()
// {
//   //Do H_core part
//   size_t k_AO = 0;
//   if (eval_OVmat(m_molbasis, S_mat) != 0)
//   {
//     std::cerr << "Warn! Overlap matrix evaluation is failed." << std::endl;
//     return 1;
//   }
//   if (eval_OVmat(m_molbasis, H_core) != 0)
//   {
//     std::cerr << "Warn! H_core matrix evaluation is failed." << std::endl;
//     return 1;
//   }

//   // H_core.print("H_core");

//   return 0;
// }


// int USCF::run()
// {
//   arma::mat Fa = H_core + Ga;
//   arma::mat Fb = H_core + Gb;
//   arma::mat Pa_old, Pb_old;
//   size_t k = 0;
//   for (; k < max_iter; k++)
//   {
//     std::cout << "Iteration: " << k << std::endl;
//     // Pa.print("Pa");
//     // Ga.print("Ga");
//     Fa.print("Fa");
//     Fb.print("Fb");
//     Pa_old = Pa;
//     Pb_old = Pb;
//     arma::eig_sym(Ea, Ca, Fa);
//     arma::eig_sym(Eb, Cb, Fb);
//     std::cout << "after solving eigen equation: " << k << std::endl;
//     Ca.print("Ca");
//     Cb.print("Cb");
//     Pa = Ca.cols(0, p - 1) * Ca.cols(0, p - 1).t();
//     if (q > 0)
//       Pb = Cb.cols(0, q - 1) * Cb.cols(0, q - 1).t();
//     else
//       Pb.zeros();
//     if (arma::approx_equal(Pa, Pa_old, "absdiff", tol) && arma::approx_equal(Pb, Pb_old, "absdiff", tol))
//       break;
//     Pa.print("Pa_new");
//     Pb.print("Pb_new");
//     Fa = H_core + Ga;
//     Fb = H_core + Gb;
//   }
//   if (k == max_iter)
//   {
//     std::cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
//     return 1;
//   }
//   Ea.print("Ea");
//   Eb.print("Eb");
//   Ca.print("Ca");
//   Cb.print("Cb");
//   return 0;
// }

// double USCF::getEnergy()
// {
//   arma::mat Ptotal = Pa + Pb;
//   Ee = arma::dot(Pa, Ga) / 2. + arma::dot(Pb, Gb) / 2.;
//   Ee += arma::dot(Ptotal, H_core);
//   Etotal = Ee + En;
//   std::cout << "Nuclear Repulsion Energy is " << En << " eV." << std::endl;
//   std::cout << "Electron Energy is " << Ee << " eV." << std::endl;
//   return Etotal;
// }
