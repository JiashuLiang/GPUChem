#include "HartreeFock_Rys_gpu.cuh"
#include "hcore.cuh"
#include "JKmat.cuh"
#include <filesystem>


__global__ void sayHello(){
  printf("Hello from GPU!\n");
}

__global__ void printAOR(double * R ){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d R( %1.2f, %1.2f, %1.2f)\n", id , R[id*3 + 0], R[id*3 + 1], R[id*3 + 2]);
}
__global__ void printEffectivecharge(int * R ){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d Effectivecharge %d\n", id , R[id]);
}

__global__ void printmAOs(AOGPU* mAOs){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d info, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x, alpha:( %1.2f, %1.2f, %1.2f), dcoef( %1.2f, %1.2f, %1.2f)\n", id,
        mAOs[id].R0[0], mAOs[id].R0[1], mAOs[id].R0[2], mAOs[id].lmn[0], mAOs[id].lmn[1], mAOs[id].lmn[2],
        mAOs[id].alpha[0], mAOs[id].alpha[1], mAOs[id].alpha[2], mAOs[id].d_coe[0], mAOs[id].d_coe[1], mAOs[id].d_coe[2]);
}


int HartreeFock_Rys_gpu::init(){
	
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



int HartreeFock_Rys_gpu::eval_OV(arma::mat &OV_mat){

    return eval_OVmat(m_molbasis, OV_mat);
}

int HartreeFock_Rys_gpu::eval_Hcore(arma::mat &H_mat){
    //m_molbasis.PrintAll();
    
    //std::cout<< "\nNow print on GPU\n";

    //printAOR<<<2,2>>>(m_molbasis_gpu.Atom_coords);
    //printEffectivecharge<<<2,2>>>(m_molbasis_gpu.effective_charges);
    //printmAOs<<<2,6>>>(m_molbasis_gpu.mAOs);
  // evaluate the H core matrix (one-electron part)
    return eval_Hcoremat(m_molbasis, H_mat);
}

int HartreeFock_Rys_gpu::eval_G(arma::mat &P_mat, arma::mat &G_mat){
  // evaluate the G matrix (two-electron part)
	return eval_Gmat_RSCF(m_molbasis, rys_root, Schwarz_mat, shreshold, P_mat, G_mat);
}

int HartreeFock_Rys_gpu::eval_J(arma::mat &P_mat, arma::mat &J_mat){
	return eval_Jmat_RSCF(m_molbasis, rys_root, Schwarz_mat, shreshold, P_mat, J_mat);
}
int HartreeFock_Rys_gpu::eval_K(arma::mat &P_mat, arma::mat &K_mat){
	return eval_Kmat_RSCF(m_molbasis, rys_root, Schwarz_mat, shreshold, P_mat, K_mat);
}