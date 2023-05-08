#include "RSCF.h"
#include <armadillo>
#include <iostream>
#include <cmath>


RSCF_DIIS::RSCF_DIIS(RSCF *m_scf_i, int max_it, double tolerence, int DIIS_circle_i): 
    m_scf(m_scf_i), max_iter(max_it), tol(tolerence), DIIS_circle(DIIS_circle_i){
    m_scf->m_scf_algorithm = this;
}


int RSCF_DIIS::init()
{

  // Initial guess for Pa use Ca = I
  m_scf->Ca.eye();
  m_scf->UpdateDensity();
  res_error = 1.;

  return 0;
}


void RSCF_DIIS::DIIS(arma::mat &e, arma::vec &c){
  int rank = c.n_elem;
  assert(e.n_cols == rank - 1);
  arma::mat e_mat (rank, rank);
  arma::vec b =arma::zeros(rank);
  b(rank-1) = -1.;
  e_mat.col(rank -1).fill(-1.);
  e_mat.row(rank -1).fill(-1.);
  e_mat(rank -1, rank -1) = 0.;
  e_mat.submat( 0, 0, rank -2, rank -2) = e.t() *e;
  // e_mat.print("e_mat");
  c =arma::solve(e_mat, b);
}

int RSCF_DIIS::run()
{
  size_t dim = m_scf->nbasis;

  m_scf->UpdateFock();
  m_scf->UpdateEnergy();
  double E_old = m_scf->Ee;

  arma::mat Pa_old(dim, dim), Fa_p(dim, dim);
  arma::mat Fa_record(dim*dim, DIIS_circle);
  arma::mat ea(dim*dim, DIIS_circle);
  arma::vec ca(DIIS_circle + 1);
  size_t k = 0;
  for (; k < max_iter; k++)
  {
    int k_DIIS = k % DIIS_circle;
    if(k_DIIS == 0 && k > 0){
      DIIS(ea, ca);
      // ca.print("ca");
      arma::vec Fa_vec(m_scf->Fa.memptr(), dim*dim, false, true);
      Fa_vec = Fa_record.col(0) *ca(0);
      for(size_t j = 1; j< DIIS_circle; j++){
        Fa_vec += Fa_record.col(j) *ca(j);
      }
    }

    // One SCF iteration
    Pa_old = m_scf->Pa;
    Fa_p = m_scf->X_mat.t() * m_scf->Fa * m_scf->X_mat; // Get Fa' = X_mat^(-1) * Fa * X_mat^(-1)
    arma::eig_sym(m_scf->Ea, m_scf->Ca, Fa_p);  // Solve eigen equation Fa' * Ca' = Ea * Ca'
    m_scf->Ca = m_scf->X_mat * m_scf->Ca; // Get Ca = X_mat * Ca'
    m_scf->UpdateDensity();
    E_old = m_scf->Ee;
    m_scf->UpdateEnergy();    
    m_scf->UpdateFock();
    arma::mat Fa_r(Fa_record.colptr(k_DIIS), dim, dim, false, true);
    Fa_r = m_scf->Fa; 
    arma::mat ea_r(ea.colptr(k_DIIS), dim, dim, false, true);
    ea_r = Fa_r * m_scf->Pa * m_scf->S_mat - m_scf->S_mat * m_scf->Pa * Fa_r;
    res_error = arma::norm(ea_r, "fro") ;
    
    std::cout << "Iteration " << k << ": Ee = " << m_scf->Ee 
        << ", Ee diff = " << std::abs(m_scf->Ee -E_old) << ", DIIS error = " << res_error << std::endl;
    if (res_error < 100 * tol && std::abs(m_scf->Ee -E_old) < tol)
      break;
  }
  if (k == max_iter)
  {
    std::cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
    return 1;
  }
  // m_scf->Ea.print("Ea");
  // m_scf->Ca.raw_print("Ca");

  return 0;
}
