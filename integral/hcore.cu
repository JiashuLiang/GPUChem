#define _USE_MATH_DEFINES
#include "hcore.cuh"
#include <basis/molecule_basis.cuh>

// #include <basis/util.h>
#include <armadillo>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>


#define NUM_THREADS 256
#define NUM_BLOCKS (nbsf * nbsf /NUM_THREADS)+ 1;

#define max(a,b) ((a)>(b)?(a):(b))
#define CHECK_TID if (TID >= num_mu*num_nu) return;
#define TID (threadIdx.x + blockIdx.x * blockDim.x)

namespace hcore_gpu{

__global__ void printmAOs(AOGPU* mAOs){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d info, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x, alpha:( %1.2f, %1.2f, %1.2f), dcoef( %1.2f, %1.2f, %1.2f)\n", id,
        mAOs[id].R0[0], mAOs[id].R0[1], mAOs[id].R0[2], mAOs[id].lmn[0], mAOs[id].lmn[1], mAOs[id].lmn[2],
        mAOs[id].alpha[0], mAOs[id].alpha[1], mAOs[id].alpha[2], mAOs[id].d_coe[0], mAOs[id].d_coe[1], mAOs[id].d_coe[2]);
}
__global__ void sayHello(){
    printf("Hello from GPU!\n");
}



void copy_AOs_to_gpu(std::vector<AO>& cpu_AOs, AOGPU* ao_gpu_array) {
    int num_ao = cpu_AOs.size();

    // Allocate memory on the GPU for the mAOs array
    cudaMalloc((void**)&ao_gpu_array, num_ao * sizeof(AOGPU));
    // Allocate and copy AO data to the GPU
    for (size_t i = 0; i < num_ao; i++) {
        AOGPU ao_gpu;

        // Set length of AOGPU
        ao_gpu.len = cpu_AOs[i].len;
        
        // Allocate memory on the GPU
        cudaMalloc((void**)&ao_gpu.R0, 3 * sizeof(double));
        cudaMalloc((void**)&ao_gpu.lmn, 3 * sizeof(unsigned int));
        cudaMalloc((void**)&ao_gpu.alpha, ao_gpu.len * sizeof(double));
        cudaMalloc((void**)&ao_gpu.d_coe, ao_gpu.len * sizeof(double));

        // Copy the data from the CPU to the GPU
        cudaMemcpy(ao_gpu.R0, cpu_AOs[i].R0.memptr(), 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.lmn, cpu_AOs[i].lmn.memptr(), 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.alpha, cpu_AOs[i].alpha.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.d_coe, cpu_AOs[i].d_coe.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);

        // Copy the AOGPU object to the mAOs array on the GPU
        cudaMemcpy(ao_gpu_array + i, &ao_gpu, sizeof(AOGPU), cudaMemcpyHostToDevice);
    }
 
}

int eval_OVmat(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &S_mat){
    const size_t nbsf = system.num_ao;
    S_mat.set_size(nbsf,nbsf);
    S_mat.zeros();

    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices;
    
    // Sort AOs by type. First, copy AOs from the device
    std::cout<< "\n Copying AOs to CPU..\n" <<std::flush;
    // AOGPU *unsorted_AOs = new AOGPU[nbsf];
    // cudaMemcpy(unsorted_AOs, system.mAOs, nbsf*sizeof(AOGPU), cudaMemcpyDeviceToHost);
    // AOGPU *unsorted_AOs = system_cpu.mAOs;
    std::cout<< "\n Sorting AOs..\n"<<std::flush;

    size_t p_start_ind = sort_AOs(system_cpu.mAOs, sorted_AOs, sorted_indices);

    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    // Perform construction of H, sorted into blocks of ss, sp, ps,pp
    
    // Copy Sorted AOs. In hindsight, probably unnecessary and couldve just copied via construct_TV;
    std::cout<< "\n Sending AOs to GPU..\n"<<std::flush;

    // AOGPU *mAO_array_gpu = (AOGPU*)malloc(nbsf*sizeof(AOGPU));
    // memset(A_coords, 0.0, Imax*sizeof(double));
    AOGPU* mAO_array_gpu;
    copy_AOs_to_gpu(sorted_AOs,mAO_array_gpu);

    // Copy Smatrices
    double *S_mat_ptr = S_mat.memptr();
    double *S_mat_gpu;

    cudaMalloc((void**)&S_mat_gpu, nbsf * nbsf* sizeof(double));
    cudaMemcpy(S_mat_gpu, S_mat_ptr, nbsf * nbsf * sizeof(double), cudaMemcpyHostToDevice);

    // MARKER
    // Perform construction of S, sorted into blocks of ss, sp, ps,pp
    int num_blocks = (nbsf * nbsf /NUM_THREADS)+ 1;
    // construct_S(S_mat, sorted_AOs, p_start_ind);
    construct_S<<<num_blocks,NUM_THREADS>>>(S_mat_gpu, mAO_array_gpu, nbsf, p_start_ind);
    // return S_mat to its original order.
    S_mat = S_mat(undo_sorted_indices, undo_sorted_indices);

    return 0;
}


int eval_Hcoremat(Molecule_basisGPU& system, Molecule_basis& system_cpu,arma::mat &H_mat){
    const size_t nbsf = system.num_ao;
    printf("nbsf is  %d\n", nbsf);
    H_mat.set_size(nbsf,nbsf);
    // arma::mat T_mat(nbsf,nbsf), V_mat(nbsf,nbsf);
    
    // T_mat.zeros();
    // V_mat.zeros();

    // std::vector<AOGPU> sorted_AOs;
    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices;
    
    // Sort AOs by type. First, copy AOs from the device
    // std::cout<< "\n Copying AOs to CPU..\n" <<std::flush;
    // AOGPU *unsorted_AOs = new AOGPU[nbsf];
    // cudaMemcpy(unsorted_AOs, system.mAOs, nbsf*sizeof(AOGPU), cudaMemcpyDeviceToHost);
    // AOGPU *unsorted_AOs = system_cpu.mAOs;
    std::cout<< "\n Sorting AOs..\n"<<std::flush;

    size_t p_start_ind = sort_AOs(system_cpu.mAOs, sorted_AOs, sorted_indices);

    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    // Perform construction of H, sorted into blocks of ss, sp, ps,pp
    
    // Copy Sorted AOs. In hindsight, probably unnecessary and couldve just copied via construct_TV;
    std::cout<< "\n Sending AOs to GPU..\n"<<std::flush;
    AOGPU* mAO_array_gpu;
    // copy_AOs_to_gpu(sorted_AOs,mAO_array_gpu);


    int num_ao = sorted_AOs.size();
    // Allocate memory on the GPU for the mAOs array
    cudaMalloc((void**)&mAO_array_gpu, num_ao * sizeof(AOGPU));
    // Allocate and copy AO data to the GPU
    for (size_t i = 0; i < num_ao; i++) {
        AOGPU ao_gpu;

        // Set length of AOGPU
        ao_gpu.len = sorted_AOs[i].len;
        
        // Allocate memory on the GPU
        cudaMalloc((void**)&ao_gpu.R0, 3 * sizeof(double));
        cudaMalloc((void**)&ao_gpu.lmn, 3 * sizeof(unsigned int));
        cudaMalloc((void**)&ao_gpu.alpha, ao_gpu.len * sizeof(double));
        cudaMalloc((void**)&ao_gpu.d_coe, ao_gpu.len * sizeof(double));

        // Copy the data from the CPU to the GPU
        cudaMemcpy(ao_gpu.R0, sorted_AOs[i].R0.memptr(), 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.lmn, sorted_AOs[i].lmn.memptr(), 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.alpha, sorted_AOs[i].alpha.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.d_coe, sorted_AOs[i].d_coe.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);

        // Copy the AOGPU object to the mAOs array on the GPU
        cudaMemcpy(mAO_array_gpu + i, &ao_gpu, sizeof(AOGPU), cudaMemcpyHostToDevice);
    }




    printmAOs<<<2,6>>>(mAO_array_gpu);
    cudaDeviceSynchronize();
    // sayHello<<<2,6>>>();
    // Copy T and V matrices
    double *T_mat_ptr = new double[nbsf*nbsf];
    double *V_mat_ptr = new double[nbsf*nbsf]; //= V_mat.memptr();
    double *T_mat_gpu, *V_mat_gpu;

    cudaMalloc((void**)&T_mat_gpu, nbsf * nbsf* sizeof(double));
    cudaMemcpy(T_mat_gpu, T_mat_ptr, nbsf * nbsf * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&V_mat_gpu, nbsf * nbsf* sizeof(double));
    cudaMemcpy(V_mat_gpu, V_mat_ptr, nbsf * nbsf * sizeof(double), cudaMemcpyHostToDevice);
    
    int num_blocks = (nbsf * nbsf /NUM_THREADS)+ 1;
    std::cout<< "\n Constructing TV, num_blocks is "<< num_blocks <<std::endl;

    construct_TV<<<num_blocks,NUM_THREADS>>>(T_mat_gpu,  V_mat_gpu, mAO_array_gpu, nbsf, p_start_ind, &system);
    cudaDeviceSynchronize();
    cudaMemcpy(T_mat_ptr, T_mat_gpu,  nbsf * nbsf * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_mat_ptr, V_mat_gpu, nbsf * nbsf * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout<< "\n Tmat matrix first element is " << T_mat_ptr[0]<<std::endl;
    arma::mat T_mat(T_mat_ptr,nbsf,nbsf,false,true);
    arma::mat V_mat(V_mat_ptr,nbsf,nbsf,false,true);
    H_mat = T_mat + V_mat;

    T_mat.print("T_mat");
    H_mat.print("H_mat");
    V_mat.print("V_mat");
    H_mat = H_mat(undo_sorted_indices, undo_sorted_indices);

    return 0;
}

size_t sort_AOs(std::vector<AO> &unsorted_AOs, std::vector<AO> &sorted_AOs, arma::uvec &sorted_indices){
    // sorts AOs, s orbitals first then p orbitals next.
    // input: unsorted_AOs
    // output: sorted_AOs, sorted_indices
    // returns: length of the s_orbs, which is also the first index of the p orbitals

    std::vector<AO> s_orbs, p_orbs;
    std::vector<size_t> s_orbs_ind, p_orbs_ind;
    for (size_t mu = 0; mu < unsorted_AOs.size(); mu++){
        int l_total = unsorted_AOs[mu].lmn(0) + unsorted_AOs[mu].lmn(1) + unsorted_AOs[mu].lmn(2);
        if (l_total == 0){
            s_orbs.push_back(unsorted_AOs[mu]);
            s_orbs_ind.push_back(mu);
        } else if (l_total == 1) {
            p_orbs.push_back(unsorted_AOs[mu]);
            p_orbs_ind.push_back(mu);
        } else {
            throw std::runtime_error("Unsupported l_total");
        }
    }
    assert(s_orbs.size() + p_orbs.size() == unsorted_AOs.size());
    size_t s_orbs_len = s_orbs.size();
    s_orbs.insert(s_orbs.end(), p_orbs.begin(), p_orbs.end()); // append p_orbs to s_orbs
    s_orbs_ind.insert(s_orbs_ind.end(), p_orbs_ind.begin(), p_orbs_ind.end());
    
    sorted_AOs = s_orbs;
    //convert s_orbs_ind to sorted_indices
    sorted_indices.set_size(s_orbs_ind.size());
    for (size_t mu = 0; mu < s_orbs_ind.size(); mu++){
        sorted_indices(mu) = s_orbs_ind[mu];
    }
    
    return s_orbs_len;
}


size_t sort_AOs(AOGPU* unsorted_AOs, const int nbsf, std::vector<AOGPU> &sorted_AOs, arma::uvec &sorted_indices){
    // sorts AOs, s orbitals first then p orbitals next. This is the AOGPU overload
    // input: unsorted_AOs, nbsf
    // output: sorted_AOs, sorted_indices
    // returns: length of the s_orbs, which is also the first index of the p orbitals

    std::vector<AOGPU> s_orbs, p_orbs;
    std::vector<size_t> s_orbs_ind, p_orbs_ind;
    std::cout<< "\n sort_AOs: Beginning mu iterations\n"<<std::flush;
    for (size_t mu = 0; mu < nbsf; mu++){
        std::cout<< "\n sort_AOs: at mu="<< mu <<"\n"<<std::flush;
        int l_total = unsorted_AOs[mu].lmn[0] + unsorted_AOs[mu].lmn[1] + unsorted_AOs[mu].lmn[2];
        std::cout<< "\n sort_AOs: finished calculating l_total"<<std::flush;
        if (l_total == 0){
            s_orbs.push_back(unsorted_AOs[mu]);
            s_orbs_ind.push_back(mu);
        } else if (l_total == 1) {
            p_orbs.push_back(unsorted_AOs[mu]);
            p_orbs_ind.push_back(mu);
        } else {
            throw std::runtime_error("Unsupported l_total");
        }
    }
     std::cout<< "\n sort_AOs: Done with Mu iterations\n"<<std::flush;
    assert(s_orbs.size() + p_orbs.size() == nbsf);
    s_orbs.insert(s_orbs.end(), p_orbs.begin(), p_orbs.end()); // append p_orbs to s_orbs
    s_orbs_ind.insert(s_orbs_ind.end(), p_orbs_ind.begin(), p_orbs_ind.end());
    
    sorted_AOs = s_orbs;
    //convert s_orbs_ind to sorted_indices
    sorted_indices.set_size(s_orbs_ind.size());
    
    for (size_t mu = 0; mu < s_orbs_ind.size(); mu++){
        sorted_indices(mu) = s_orbs_ind[mu];
    }
    return s_orbs.size();
}


__device__ void construct_S_block(double* Smat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid){
    // size_t tid = TID;
    // CHECK_TID // return if thread num exceeds num of elements in block
    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;

    // arma::mat Tmat_block(Tmat + mu_start_ind + nu_start_ind*nbsf, num_mu, num_nu, false, true);
    // Tmat_block(mu,nu) = eval_Tmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);
    
    Smat[mu_nu_ind] = eval_Smunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);


}
__global__ void construct_S(double* Smat,  AOGPU* mAOs, size_t nbsf, size_t p_start_ind){
    size_t tid = TID;

    size_t p_dim = nbsf - p_start_ind;

    construct_S_block(Smat, mAOs, 0, 0, p_start_ind, p_start_ind, nbsf, tid); // ss
    construct_S_block(Smat, mAOs, p_start_ind, 0, p_dim, p_start_ind, nbsf, tid); // ps
    construct_S_block(Smat, mAOs, 0, p_start_ind, p_start_ind, p_dim, nbsf, tid); // sp
    construct_S_block(Smat, mAOs, p_start_ind, p_start_ind, p_dim, p_dim, nbsf, tid); // pp

}



__device__ void construct_T_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid){
    // size_t tid = TID;
    // CHECK_TID // return if thread num exceeds num of elements in block
    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;

    // arma::mat Tmat_block(Tmat + mu_start_ind + nu_start_ind*nbsf, num_mu, num_nu, false, true);
    // Tmat_block(mu,nu) = eval_Tmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);
    int id = mu + mu_start_ind;
    // printf("tid %d computing for mu= %d, nu=%d and mu_nu_ind=%d\n", int(tid), int(mu + mu_start_ind), int(nu + nu_start_ind), int(mu_nu_ind));
    // printf("%d info, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x, alpha:( %1.2f, %1.2f, %1.2f), dcoef( %1.2f, %1.2f, %1.2f) and len %d\n", int(tid),
    //     mAOs[id].R0[0], mAOs[id].R0[1], mAOs[id].R0[2], mAOs[id].lmn[0], mAOs[id].lmn[1], mAOs[id].lmn[2],
    //     mAOs[id].alpha[0], mAOs[id].alpha[1], mAOs[id].alpha[2], mAOs[id].d_coe[0], mAOs[id].d_coe[1], mAOs[id].d_coe[2], mAOs[id].len);

    Tmat[mu_nu_ind] = eval_Tmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);

    // printf("tid %d computes that Tmat[%d]= %1.2f\n", int(tid), int(mu_nu_ind), Tmat[mu_nu_ind]);

}

__device__ void construct_V_block(double* Vmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, Molecule_basisGPU* mol, size_t tid){
    // size_t tid = TID;
    // CHECK_TID // return if thread num exceeds num of elements in block
    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;


    // arma::mat Vmat_block(Vmat + mu_start_ind + nu_start_ind*nbsf, num_mu, num_nu, false, true);
    // Vmat_block(mu,nu) = eval_Vmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind], mol);
    printf("tid %d computing for mu= %d, nu=%d and mu_nu_ind=%d\n", int(tid), int(mu), int(nu), int(mu_nu_ind));
    Vmat[mu_nu_ind] = eval_Vmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind], mol);
    printf("tid %d computes that Vmat[%d]= %1.2f\n", int(tid), int(mu_nu_ind),Vmat[mu_nu_ind]);

}


__global__ void construct_TV(double* Tmat, double* Vmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind, Molecule_basisGPU* mol){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t p_dim = nbsf - p_start_ind;
    if (tid==0) printf("construct_TV beginning, p_dim is  %d, nbsf is %d \n", int(p_dim), int(nbsf));

    // construct_T_block(Tmat, mAOs, 0,           0,           p_start_ind,  p_start_ind, nbsf, tid); // ss
    // construct_T_block(Tmat, mAOs, p_start_ind, 0,           p_dim,        p_start_ind, nbsf, tid); // ps
    
    // construct_T_block(Tmat, mAOs, 0          , p_start_ind, p_start_ind,  p_dim, nbsf, tid); // sp

    // construct_T_block(Tmat, mAOs, p_start_ind, p_start_ind, p_dim,        p_dim, nbsf, tid); // pp
    construct_T_block(Tmat, mAOs, 0,           0,           nbsf,  nbsf, nbsf, tid); // ss
    
    if (tid==0) printf("construct_T is done, tid %d\n", int(tid));
    if (tid==0) printf(" Tmat[0]= %1.2f\n",Tmat[0]);

    // construct_V_block(Vmat, mAOs, 0, 0, p_start_ind, p_start_ind, nbsf, mol, tid); // ss
    // construct_V_block(Vmat, mAOs, p_start_ind, 0, p_dim, p_start_ind, nbsf, mol, tid); // ps
    // construct_V_block(Vmat, mAOs, 0, p_start_ind, p_start_ind, p_dim, nbsf, mol, tid); // sp
    // construct_V_block(Vmat, mAOs, p_start_ind, p_start_ind, p_dim, p_dim, nbsf, mol, tid); // pp
    construct_V_block(Vmat, mAOs, 0,           0,           nbsf,  nbsf, nbsf, mol, tid); // ss
    // printf("construct_T is done, tid %d\n", int(tid));

}


__device__ double eval_Smunu(AOGPU &mu, AOGPU &nu){
    // assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?
    
    size_t mu_no_primitives = mu.len;
    size_t nu_no_primitives = nu.len;
    
    double total = 0.0;
    int l1 = mu.lmn[0];
    int m1 = mu.lmn[1];
    int n1 = mu.lmn[2];
    
    int l2 = nu.lmn[0];
    int m2 = nu.lmn[1];
    int n2 = nu.lmn[2];
    double* A = mu.R0; // pointer
    double* B = nu.R0;

    for (size_t mup = 0; mup < mu_no_primitives; mup++){
        double alpha = mu.alpha[mup];
        double d_kmu = mu.d_coe[mup];

        for (size_t nup = 0; nup < nu_no_primitives; nup++){
            double beta = nu.alpha[nup];
            double d_knu = nu.d_coe[nup];
            total +=  d_knu * d_kmu * overlap(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta);
        }
    }
    return total;

}


__device__ double eval_Tmunu(AOGPU &mu, AOGPU &nu){
    // assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?


    size_t mu_no_primitives = mu.len;
    size_t nu_no_primitives = nu.len;
        // printf("%d info mu_no_primitives %d", int(tid), int(mu_no_primitives) );
    double total = 0.0;
    int l1 = mu.lmn[0];
    int m1 = mu.lmn[1];
    int n1 = mu.lmn[2];
    
    int l2 = nu.lmn[0];
    int m2 = nu.lmn[1];
    int n2 = nu.lmn[2];
    double* A = mu.R0; // pointer
    double* B = nu.R0;
    // printf("Reached start of for-loops\n");
    for (size_t mup = 0; mup < mu_no_primitives; mup++){
        double alpha = mu.alpha[mup];
        double d_kmu = mu.d_coe[mup];
        // printf("Reached start of inner for-loops\n");
        for (size_t nup = 0; nup < nu_no_primitives; nup++){
            double beta = nu.alpha[nup];
            double d_knu = nu.d_coe[nup];
            // printf("Reached start of kinetic\n");
            total +=  d_knu * d_kmu * kinetic(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta);
        }
    }
    return total;
}



__device__ double eval_Vmunu(AOGPU &mu, AOGPU &nu, const Molecule_basisGPU* mol){
    // nuclear attraction
    // assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?
    
    size_t mu_no_primitives = mu.len;
    size_t nu_no_primitives = nu.len;
    
    double total = 0.0;
    int l1 = mu.lmn[0];
    int m1 = mu.lmn[1];
    int n1 = mu.lmn[2];
    
    int l2 = nu.lmn[0];
    int m2 = nu.lmn[1];
    int n2 = nu.lmn[2];
    double* A = mu.R0; // pointer
    double* B = nu.R0;
    for (size_t c = 0; c < (*mol).num_atom; c++){
        // arma::vec C = mol.mAtoms[c].m_coord; // coordinates of the atom
        double* C = (*mol).Atom_coords + 3*c; // pointer arithmetic to get starting point of C coords
        int Z = (*mol).effective_charges[c];
        for (size_t mup = 0; mup < mu_no_primitives; mup++){
            double alpha = mu.alpha[mup];
            double d_kmu = mu.d_coe[mup];

            for (size_t nup = 0; nup < nu_no_primitives; nup++){
                double beta = nu.alpha[nup];
                double d_knu = nu.d_coe[nup];
                total +=  d_knu * d_kmu * Z* nuclear_attraction(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta, C);
            }
        }
    }
    return total;
}


__device__ double gammln(double xx){
    // From Numerical Recipes, 6.1 and 6.2
    // Returns the value ln[Γ(xx)] for xx > 0.
    // Internal arithmetic will be done in double precision, a nicety that you can omit if five-figure accuracy is good enough.
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146,-86.50532032941677,
    24.01409824083091,-1.231739572450155,
    0.1208650973866179e-2,-0.5395239384953e-5};
    int j;
    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) ser += cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
}

__device__ void gser(double *gamser, double a, double x, double *gln) {
    // From Numerical Recipes, 6.1 and 6.2
    // Returns the incomplete gamma function P (a, x) evaluated by its series representation as gamser Also returns ln Γ(a) as gln.

    #define ITMAX 100 
    #define EPS 3.0e-7 
    // double gammln(double xx);
    // void nrerror(char error_text[]);
    int n;
    double sum,del,ap;
    *gln=gammln(a);
    if (x <= 0.0) {
        assert(x >= 0.0);
        // if (x < 0.0) throw std::runtime_error("x less than 0 in routine gser");
        *gamser=0.0;
        return;
    } else {
        ap=a;
        del=sum=1.0/a;
        for (n=1;n<=ITMAX;n++) {
            ++ap;
            del *= x/ap;
            sum += del;
            if (fabs(del) < fabs(sum)*EPS) {
                *gamser=sum*exp(-x+a*log(x)-(*gln));
                return;
            }
        }
        assert(false);
        // throw std::runtime_error("a too large, ITMAX too small in routine gser");
        return;
    }
}

__device__ void gcf(double *gammcf, double a, double x, double *gln){
    // From Numerical Recipes, 6.1 and 6.2
    // Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation as gammcf. Also returns ln Γ(a) as gln.
    // double gammln(double xx);
    // void nrerror(char error_text[]);
    #define ITMAX 100 
    #define EPS 3.0e-7 
    #define FPMIN 1.0e-30 

    int i;
    
    double an,b,c,d,del,h;

    *gln = gammln(a);
    b = x + 1.0 - a;  //Set up for evaluating continued fractionby modified Lentz’s method (§5.2) with b0 = 0.
    c = 1.0/FPMIN;
    d = 1.0/b;
    h=d;
    for (i=1;i<=ITMAX;i++) { // Iterate to convergence.
        an = -i*(i-a);
        b += 2.0;
        d = an*d + b;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = b + an/c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0/d;
        del = d*c;
        h *= del;
        if (fabs(del-1.0) < EPS) break;
    }
    // if (i > ITMAX) throw std::runtime_error("a too large, ITMAX too small in gcf");
    assert (i <= ITMAX);
    *gammcf = exp(-x+a*log(x)-(*gln))*h; //Put factors in front.
}

__device__ double gammp(double a, double x){
    // Returns the incomplete gamma function P (a, x). From Numerical Recipes, section 6.1 and 6.2
    double gam, gamc, gln;
    // if (x < 0.0 || a <= 0.0) throw std::runtime_error("Invalid arguments in routine gammp");
    assert(!(x < 0.0 || a <= 0.0));
    if (x < (a+1.0)) {// Use the series representation.
        gser(&gam,a,x,&gln);
    } else { //Use the continued fraction representation
        gcf(&gamc,a,x,&gln);
        gam = 1-gamc;
        
    }
    return exp(gln)*gam;
}

__device__ double Fgamma(int m, double x){
    // Incomplete Gamma Function
    double SMALL=1e-12;
    double m_d = (double) m; // convert to double explicitly, prolly notneeded
    x = max(x,SMALL);
    // std::cout<<"-m_d-0.5 --" <<(-m_d-0.5)<<std::endl;
    // std::cout<<"-m-0.5 --" <<(-m-0.5)<<std::endl;
    return 0.5*pow(x,-m_d-0.5)*gammp(m_d+0.5,x);
}

__device__ int factorial (int n){
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

__device__ double DoubleFactorial(int n)
{
  if (n < -1){
    printf("DoubleFactorial: Input should be 1 at least\n");
    assert(false);
  }
  if (n == 0 || n == -1)
    return 1;
  double result = 1;
  while (n > 1)
  {
    result *= n;
    n -= 2;
  }
  return result;
}


__device__ int nCk (int n, int k){
    return factorial(n)/(factorial(k) * factorial(n-k));
}

__device__ vector_gpu gaussian_product_center(double alpha, vector_gpu &A, double beta, vector_gpu &B){
    //computes the new gaussian center P.
    // double* P;
    return (A*alpha + B*beta)*(1.0/(alpha + beta));

    // for (size_t i =0; i<3; i++){
    //     P[i] = (alpha*A[i]+ beta*B[i])/(alpha + beta);
    // }
    // return P;
}

__device__ double poly_binom_expans_terms(int n, int ia, int ib, double PminusA_1d, double PminusB_1d){
    // computes the binomial expansion for the terms where ia + ib = n.
    double total = 0.0;

    for (int t = 0; t < n + 1; t++){
        if (n - ia <= t && t <= ib){
            total += nCk(ia, n-t) * nCk (ib,t) * pow(PminusA_1d, ia-n+t) * pow(PminusB_1d, ib-t);
        }
    }
    return total;
}

__device__ double overlap_1d(int l1, int l2, double PminusA_1d, double PminusB_1d, double gamma){
    double total  = 0.0;
    // for (int i = 0; i < (1+ int(floor((l1+l2)/2))); i++){
    for (int i = 0; i < (1+ int(floorf(l1+l2)/2)); i++){
        total += poly_binom_expans_terms(2*i, l1, l2,PminusA_1d, PminusB_1d )* DoubleFactorial((2*i-1)/pow(2*gamma, i));
    }
    return total;

}

__device__ double overlap(double* A,  int l1, int m1, int n1,double alpha, double* B, int l2, int m2, int n2,double beta ){
    double gamma = alpha + beta;
    // printf("started overlap\n");
    vector_gpu A_vec(A,3);
    vector_gpu B_vec(B,3);
    // printf("Finished def A and B\n");
    vector_gpu P_vec = gaussian_product_center(alpha, A_vec, beta, B_vec);
    // printf("Finished gaussian_product_center\n");
    double* P = P_vec.coords();
    
    double prefactor = pow(M_PI/gamma,1.5) * exp(-alpha * beta * pow((A_vec-B_vec).norm(),2)/gamma);
    // printf("Finished prefactor\n");
    // printf("Calculating overlap 1d\n");
    // printf("P( %1.2f, %1.2f, %1.2f),A:( %1.2f, %1.2f, %1.2f), B( %1.2f, %1.2f, %1.2f)\n", P[0], P[1], P[2],A[0], A[1], A[2],B[0], B[1], B[2]);
    double sx = overlap_1d(l1,l2,P[0]-A[0],P[0]-B[0],gamma);
    double sy = overlap_1d(m1,m2,P[1]-A[1],P[1]-B[1],gamma);
    double sz = overlap_1d(n1,n2,P[2]-A[2],P[2]-B[2],gamma);
    // printf("Finished overlap_1d\n");
    return prefactor * sx * sy * sz;
}

__device__ double kinetic(double* A,int l1, int m1, int n1,double alpha, double* B, int l2, int m2, int n2, double beta){
    // Formulation from JPS (21) 11, Nov 1966 by H Taketa et. al
    
    
    double term0 = beta*(2*(l2+m2+n2)+3)*overlap(A,l1,m1,n1,alpha,B,l2,m2,n2,beta);
    // printf("finished term0\n");
    double term1 = -2*pow(beta,2)*(overlap(A,l1,m1,n1,alpha, B,l2+2,m2,n2,beta) +\
                            overlap(A,l1,m1,n1,alpha, B,l2,m2+2,n2,beta) +\
                            overlap(A,l1,m1,n1,alpha, B,l2,m2,n2+2,beta));
    double term2 = -0.5*(l2*(l2-1)*overlap(A,l1,m1,n1,alpha, B,l2-2,m2,n2,beta) +\
                  m2*(m2-1)*overlap(A,l1,m1,n1,alpha, B,l2,m2-2,n2,beta) +\
                  n2*(n2-1)*overlap(A,l1,m1,n1,alpha, B,l2,m2,n2-2,beta));
    // printf("finished term2\n");
    return term0 + term1 + term2;
}



__device__ double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma){

    return pow(-1,i)*poly_binom_expans_terms(i,l1,l2,PAx,PBx)*\
           pow(-1,u)*factorial(i)*pow(CPx,i-2*r-2*u)*\
           pow(0.25/gamma,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u);
}

__device__ vector_gpu A_tensor(int l1, int l2, double PA, double PB, double CP, double g){
    int Imax = l1+l2+1;
    // double A_coords[Imax] = {0};
    double *A_coords = (double*)malloc(Imax*sizeof(double));
    memset(A_coords, 0.0, Imax*sizeof(double));

    // arma::vec A(Imax);
    // A.zeros();
    for (int i = 0; i < Imax; i++){
        for (int r = 0; r < int(floorf(i/2)+1); r++){
            for (int u = 0; u < int(floorf((i-2*r)/2)+1); u++){
                int  I = i - 2*r - u;
                A_coords[I] = A_coords[I] + A_term(i,r,u,l1,l2,PA,PB,CP,g);
            }
        }
    }
    vector_gpu A(A_coords,Imax);
    return A;

}
__device__ double nuclear_attraction(double *A,int l1, int m1, int n1,double alpha, double *B, int l2, int m2, int n2,double beta, double *C){
    // Formulation from JPS (21) 11, Nov 1966 by H Taketa et. al
    vector_gpu A_vec(A,3);
    vector_gpu B_vec(B,3);
    vector_gpu C_vec(C,3);
    double gamma = alpha + beta;

    vector_gpu P_vec = gaussian_product_center(alpha, A_vec, beta, B_vec);
    // double P* =
    // double rab2 = pow(arma::norm(A-B),2);
    // double rcp2 = pow(arma::norm(C-P),2);

    double rab2 = pow((A_vec-B_vec).norm(),2);
    double rcp2 = pow((C_vec-P_vec).norm(),2);
    
    vector_gpu dPA_vec = P_vec-A_vec;
    vector_gpu dPB_vec = P_vec-B_vec;
    vector_gpu dPC_vec = P_vec-C_vec;

    double* dPA = dPA_vec.coords();
    double* dPB = dPB_vec.coords();
    double* dPC = dPC_vec.coords();

    vector_gpu Ax_vec = A_tensor(l1,l2,dPA[0],dPB[0],dPC[0],gamma);
    vector_gpu Ay_vec = A_tensor(m1,m2,dPA[1],dPB[1],dPC[1],gamma);
    vector_gpu Az_vec = A_tensor(n1,n2,dPA[2],dPB[2],dPC[2],gamma);

    double* Ax = Ax_vec.coords();
    double* Ay = Ay_vec.coords();
    double* Az = Az_vec.coords();

    double total = 0.0;
    for (int I = 0; I < l1+l2+1; I++)
        for (int J = 0; J < m1+m2+1; J++)
            for (int K = 0; K < n1+n2+1; K++)
                total += Ax[I]*Ay[J]*Az[K]*Fgamma(I+J+K,rcp2*gamma);
                
    double val= -2*M_PI/gamma*exp(-alpha*beta*rab2/gamma)*total;
    return val;
}

} // namespace hcore_gpu