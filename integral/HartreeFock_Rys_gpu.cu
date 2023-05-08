#include "HartreeFock_Rys_gpu.cuh"
#include "hcore.h"
#include "JKmat.cuh"
#include <filesystem>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void sayHello()
{
    printf("Hello from GPU!\n");
}

__global__ void printAOR(double *R)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d R( %1.2f, %1.2f, %1.2f)\n", id, R[id * 3 + 0], R[id * 3 + 1], R[id * 3 + 2]);
}
__global__ void printEffectivecharge(int *R)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d Effectivecharge %d\n", id, R[id]);
}

__global__ void printmAOs(AOGPU *mAOs)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d info, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x, alpha:( %1.2f, %1.2f, %1.2f), dcoef( %1.2f, %1.2f, %1.2f)\n", id,
           mAOs[id].R0[0], mAOs[id].R0[1], mAOs[id].R0[2], mAOs[id].lmn[0], mAOs[id].lmn[1], mAOs[id].lmn[2],
           mAOs[id].alpha[0], mAOs[id].alpha[1], mAOs[id].alpha[2], mAOs[id].d_coe[0], mAOs[id].d_coe[1], mAOs[id].d_coe[2]);
}

HartreeFock_Rys_gpu::~HartreeFock_Rys_gpu()
{
    if (Schwarz_mat != nullptr)
        cudaFree(Schwarz_mat);
    if (rys_root != nullptr)
        cudaFree(rys_root);
}

int HartreeFock_Rys_gpu::init()
{

    size_t dim = m_molbasis.mAOs.size();

    // checking the basis set to see if there is high angular momentum stuff
    for (int mu = 0; mu < dim; mu++)
    {
        if (arma::accu(m_molbasis.mAOs[mu].lmn) >= 2)
        {
            std::cout << "higher angular momentum basis detected! Can only do s and p";
            return 1;
        }
    }

    // loading rys roots
    std::string aux;
    if (const char *env_p = std::getenv("GPUChem_aux"))
    {
        aux = std::string(env_p);
        if (!std::filesystem::is_directory(aux))
        {
            throw std::runtime_error("basis/basis_set.cpp: The directory specified by GPUChem_aux does not exist!");
        }
    }
    arma::mat rys_root_arma;
    // text file contatins rys root (squared) and their weights from X = 0 to 30 (0.01 increment)
    rys_root_arma.load(aux + "/rys_root.txt");

    // Get the size of rys_root_arma and Schwarz_mat_arma
    Schwarz_mat_dim0 = dim;
    Schwarz_mat_dim1 = dim;
    rys_root_dim0 = rys_root_arma.n_cols;
    rys_root_dim1 = rys_root_arma.n_rows;

    // Allocate memory on GPU for rys_root and Schwarz_mat
    cudaMalloc((void **)&rys_root, sizeof(double) * rys_root_arma.n_elem);
    cudaMalloc((void **)&Schwarz_mat, sizeof(double) * dim * dim);

    // Copy rys_root from CPU to GPU
    cudaMemcpy(rys_root, rys_root_arma.memptr(), sizeof(double) * rys_root_arma.n_elem, cudaMemcpyHostToDevice);

    // evaluate the Schwarz matrix
    // set dim3 grid and block to 2D
    dim3 blockDim(8, 8);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    // call the kernel
    eval_Schwarzmat_GPU<<<gridDim, blockDim>>>(m_molbasis_gpu.mAOs, rys_root, Schwarz_mat, dim, rys_root_dim1);


    return 0;
}

int HartreeFock_Rys_gpu::eval_OV(arma::mat &OV_mat)
{

    return eval_OVmat(m_molbasis, OV_mat);
}

int HartreeFock_Rys_gpu::eval_Hcore(arma::mat &H_mat)
{
    // m_molbasis.PrintAll();

    // std::cout<< "\nNow print on GPU\n";

    // printAOR<<<2,2>>>(m_molbasis_gpu.Atom_coords);
    // printEffectivecharge<<<2,2>>>(m_molbasis_gpu.effective_charges);
    // printmAOs<<<2,6>>>(m_molbasis_gpu.mAOs);
    // evaluate the H core matrix (one-electron part)
    return eval_Hcoremat(m_molbasis, H_mat);
    return 0;
}

int HartreeFock_Rys_gpu::eval_G(arma::mat &P_mat, arma::mat &G_mat)
{
    size_t dim = m_molbasis.mAOs.size();
    // check the size of P_mat and G_mat
    if (P_mat.n_rows != dim || P_mat.n_cols != dim || G_mat.n_rows != dim || G_mat.n_cols != dim)
    {
        std::cout << "P_mat or G_mat has wrong dimension!";
        return 1;
    }
    double *P_mat_gpu, *G_mat_gpu;
    cudaMalloc((void **)&P_mat_gpu, sizeof(double) * P_mat.n_elem);
    cudaMalloc((void **)&G_mat_gpu, sizeof(double) * G_mat.n_elem);
    cudaMemcpy(P_mat_gpu, P_mat.memptr(), sizeof(double) * P_mat.n_elem, cudaMemcpyHostToDevice);

    // evaluate the G matrix (two-electron part)
    int ok = 0;
    ok =  eval_Gmat_RSCF(m_molbasis_gpu, rys_root, Schwarz_mat, shreshold, P_mat_gpu, G_mat_gpu, rys_root_dim1);

    // copy G_mat from GPU to CPU
    cudaMemcpy(G_mat.memptr(), G_mat_gpu, sizeof(double) * G_mat.n_elem, cudaMemcpyDeviceToHost);

    return ok;
}

int HartreeFock_Rys_gpu::eval_J(arma::mat &P_mat, arma::mat &J_mat)
{

    size_t dim = m_molbasis.mAOs.size();
    // check the size of P_mat and J_mat
    if (P_mat.n_rows != dim || P_mat.n_cols != dim || J_mat.n_rows != dim || J_mat.n_cols != dim)
    {
        std::cout << "P_mat or J_mat has wrong dimension!";
        return 1;
    }
    double *P_mat_gpu, *J_mat_gpu;
    cudaMalloc((void **)&P_mat_gpu, sizeof(double) * P_mat.n_elem);
    cudaMalloc((void **)&J_mat_gpu, sizeof(double) * J_mat.n_elem);
    cudaMemcpy(P_mat_gpu, P_mat.memptr(), sizeof(double) * P_mat.n_elem, cudaMemcpyHostToDevice);

    
	// to find the maximum element in Schwarz_mat 
	thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(Schwarz_mat);
	thrust::device_ptr<double> max_ptr = thrust::max_element(dev_ptr, dev_ptr + dim * dim);
	double schwarz_max = *max_ptr;

    // Evaluate the J matrix
    int ok = 0;
    ok =  eval_Jmat_RSCF(m_molbasis_gpu, rys_root, Schwarz_mat, shreshold, schwarz_max, P_mat_gpu, J_mat_gpu, rys_root_dim1);

    // copy J_mat from GPU to CPU
    cudaMemcpy(J_mat.memptr(), J_mat_gpu, sizeof(double) * J_mat.n_elem, cudaMemcpyDeviceToHost);

    return ok;
}
int HartreeFock_Rys_gpu::eval_K(arma::mat &P_mat, arma::mat &K_mat)
{

    size_t dim = m_molbasis.mAOs.size();
    // check the size of P_mat and K_mat
    if (P_mat.n_rows != dim || P_mat.n_cols != dim || K_mat.n_rows != dim || K_mat.n_cols != dim)
    {
        std::cout << "P_mat or K_mat has wrong dimension!";
        return 1;
    }
    double *P_mat_gpu, *K_mat_gpu;
    cudaMalloc((void **)&P_mat_gpu, sizeof(double) * P_mat.n_elem);
    cudaMalloc((void **)&K_mat_gpu, sizeof(double) * K_mat.n_elem);
    cudaMemcpy(P_mat_gpu, P_mat.memptr(), sizeof(double) * P_mat.n_elem, cudaMemcpyHostToDevice);

	// to find the maximum element in Schwarz_mat 
	thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(Schwarz_mat);
	thrust::device_ptr<double> max_ptr = thrust::max_element(dev_ptr, dev_ptr + dim * dim);
	double schwarz_max = *max_ptr;

    // evaluate the K matrix (two-electron part)
    int ok = 0;
    ok =  eval_Kmat_RSCF(m_molbasis_gpu, rys_root, Schwarz_mat, shreshold, schwarz_max, P_mat_gpu, K_mat_gpu, rys_root_dim1);

    // copy K_mat from GPU to CPU
    cudaMemcpy(K_mat.memptr(), K_mat_gpu, sizeof(double) * K_mat.n_elem, cudaMemcpyDeviceToHost);

    return ok;
}