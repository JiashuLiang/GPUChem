#include "Hamiltonian.h"
#include "hcore.h"
#include "JKmat.h"
#include <filesystem>


int HartreeFock_Rys::init(){
	
  size_t dim = m_molbasis.mAOs.size();
  // loading rys roots
  std::string aux;
  if (const char* env_p = std::getenv("GPUChem_aux")){
    aux = std::string(env_p);
      if (!std::filesystem::is_directory(aux)) {
        throw std::runtime_error("basis/basis_set.cpp: The directory specified by GPUChem_aux does not exist!");
    }
  }
  rys_root.load(aux + "/rys_root.txt");
  // text file contatins rys root (squared) and their weights from X = 0 to 30 (0.01 increment)

  // make the matrix for Schwarz prescreening
  Schwarz_mat.set_size(dim, dim);
  Schwarz_mat.zeros();
  eval_Schwarzmat(m_molbasis, rys_root, Schwarz_mat);

  return 0;
}
int HartreeFock_Rys::eval_OV(arma::mat &OV_mat){
    return eval_OVmat(m_molbasis, OV_mat);
}

int HartreeFock_Rys::eval_Hcore(arma::mat &H_mat){
  // evaluate the H core matrix (one-electron part)
    return eval_Hcoremat(m_molbasis, H_mat);
}

int HartreeFock_Rys::eval_G(arma::mat &P_mat, arma::mat &G_mat){
  // evaluate the G matrix (two-electron part)
	return eval_Gmat_RSCF(m_molbasis, rys_root, Schwarz_mat, shreshold, P_mat, G_mat);
}