#include "RSCF.h"
#include <armadillo>
#include <iostream>
#include <cmath>


RSCF_plain::RSCF_plain(RSCF *m_scf_i, int max_it, double tolerence): m_scf(m_scf_i), max_iter(max_it), tol(tolerence){
    m_scf->m_scf_algorithm = this;
}


int RSCF_plain::init()
{

  // Initial guess for Pa use Ca = I
  m_scf->Ca.eye();
  m_scf->UpdateDensity();
  res_error = 1.;


  return 0;
}


int RSCF_plain::run()
{
  // std::cout << "SCF iteration starts." << std::endl;
  // Get initial guess for Ga
  m_scf->UpdateFock();
  arma::mat Pa_old, Fa_p;
  double E_old;
  size_t k = 0;
  std::cout << "Iteration start! "<< std::endl;
  for (; k < max_iter; k++)
  {
    // Pa.print("Pa");
    // Ga.print("Ga");
    // Fa.print("Fa");
    Pa_old = m_scf->Pa;
    Fa_p = m_scf->X_mat.t() * m_scf->Fa * m_scf->X_mat; // Get Fa' = X_mat^(-1) * Fa * X_mat^(-1)
    arma::eig_sym(m_scf->Ea, m_scf->Ca, Fa_p);  // Solve eigen equation Fa' * Ca' = Ea * Ca'
    // Fa.print("Fa'");
    // Ea.print("Ea");
    m_scf->Ca = m_scf->X_mat * m_scf->Ca; // Get Ca = X_mat * Ca'
    m_scf->UpdateDensity();
    E_old = m_scf->Ee;
    m_scf->UpdateEnergy();
    res_error = arma::norm(m_scf->Pa- Pa_old, "fro");
    std::cout << "Iteration " << k << ": the res_errorerence fro norm is " << res_error << ", Ee = " << m_scf->Ee << ", Ee  diff  = " << std::abs(m_scf->Ee -E_old) << std::endl;
    if (res_error < 100 * tol && std::abs(m_scf->Ee -E_old) < tol)
      break;
    // Pa.print("Pa_new");
    m_scf->UpdateFock();
  }
  if (k == max_iter)
  {
    std::cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
    return 1;
  }
  m_scf->Ea.raw_print("Ea");
  // m_scf->Ca.raw_print("Ca");
  return 0;
}
