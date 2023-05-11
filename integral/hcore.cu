#define _USE_MATH_DEFINES
#include "hcore.cuh"
#include <basis/molecule_basis.cuh>
#include <basis/AO.h>

// #include <basis/util.h>
#include <armadillo>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>


#define NUM_THREADS 256
#define max(a,b) ((a)>(b)?(a):(b))
#define OneDemension_threadsPerBlock 8


namespace hcore_gpu{

int eval_OVmat(Molecule_basisGPU& system, double * S_mat_gpu){
    const size_t nbsf = system.num_ao;

    // We will calculate the whole S matrix block by block
    // If The AO is sorted, this will it more possible that each (mu|nu) block have same kind of mu, nu
	dim3 blockDim(OneDemension_threadsPerBlock, OneDemension_threadsPerBlock); // OneDemension_threadsPerBlock^2 threads per block
    dim3 gridDim((nbsf + blockDim.x - 1) / blockDim.x, (nbsf + blockDim.y - 1) / blockDim.y);

    construct_S_whole_mat<<<gridDim, blockDim>>>(S_mat_gpu, system.mAOs, nbsf);
    
    return 0;
}

int eval_Hcoremat(Molecule_basisGPU& system, double * H_mat_gpu){
    const size_t nbsf = system.num_ao;
    
    // We will calculate the whole S matrix block by block
    // If The AO is sorted, this will it more possible that each (mu|H|nu) block have same kind of mu, nu
	dim3 blockDim(OneDemension_threadsPerBlock, OneDemension_threadsPerBlock); // OneDemension_threadsPerBlock^2 threads per block
    dim3 gridDim((nbsf + blockDim.x - 1) / blockDim.x, (nbsf + blockDim.y - 1) / blockDim.y);

    // Set Hmat to zero
    cudaMemset(H_mat_gpu, 0, nbsf * nbsf * sizeof(double));

    //Construct Tmat and added to Hmat (Not overwriting!!!)
    construct_T<<<gridDim, blockDim>>>(H_mat_gpu, system.mAOs, nbsf);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "construct_T launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // goto Error;
    }

    // Construct Vmat and added to Hmat (Not overwriting!!!)
    construct_V<<<gridDim, blockDim>>>(H_mat_gpu, system.mAOs, nbsf, system.Atom_coords, system.effective_charges, system.num_atom);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "construct_V launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // goto Error;
    }
    
    return 0;
}


__global__ void construct_S_whole_mat(double* Smat,  AOGPU* mAOs, size_t nbsf){
    
	int mu = blockIdx.x * blockDim.x + threadIdx.x;
	int nu = blockIdx.y * blockDim.y + threadIdx.y;

    if (mu >= nbsf || nu >= nbsf) return;
    int index = mu + nu * nbsf;
    
    Smat[index] = eval_Smunu(mAOs[mu], mAOs[nu]);
}

__global__ void construct_T(double* Tmat, AOGPU* mAOs, size_t nbsf){
	int mu = blockIdx.x * blockDim.x + threadIdx.x;
	int nu = blockIdx.y * blockDim.y + threadIdx.y;
    if (mu >= nbsf || nu >= nbsf) return;
    int index = mu + nu * nbsf;

    Tmat[index] += eval_Tmunu(mAOs[mu], mAOs[nu]);

}

__global__ void construct_V(double* Vmat, AOGPU* mAOs, size_t nbsf, double* Atom_coords, const int* effective_charges, const int num_atom){
	int mu = blockIdx.x * blockDim.x + threadIdx.x;
	int nu = blockIdx.y * blockDim.y + threadIdx.y;
    if (mu >= nbsf || nu >= nbsf) return;
    int index = mu + nu * nbsf;

    Vmat[index] += eval_Vmunu(mAOs[mu], mAOs[nu], Atom_coords, effective_charges, num_atom);
}











int eval_OVmat_sort_inside(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &S_mat){
    const size_t nbsf = system.num_ao;
    S_mat.set_size(nbsf,nbsf);
    S_mat.zeros();

    // Sort AOs as s, px, py, pz
    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices, sorted_offs;
    sort_AOs(system_cpu.mAOs, sorted_AOs, sorted_indices, sorted_offs);
    size_t p_start_ind = sorted_offs(1); // The index of the first p AO

    // get the undo sorted indices
    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    // Copy Sorted AOs. In hindsight, probably unnecessary and couldve just copied via construct_TV;
    AOGPU* mAO_array_gpu;

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


    // Allocate memory for S_mat on the GPU
    double *S_mat_gpu;
    cudaMalloc((void**)&S_mat_gpu, S_mat.n_elem * sizeof(double));
    cudaMemset(S_mat_gpu, 0.0, sizeof(double) * nbsf*nbsf);

    int num_blocks = (nbsf * nbsf /NUM_THREADS)+ 1;
    // Perform construction of S, sorted into blocks of ss, sp, ps,pp
    construct_S<<<num_blocks,NUM_THREADS>>>(S_mat_gpu, mAO_array_gpu, nbsf, p_start_ind);
    // Copy S_mat from GPU to CPU
    cudaMemcpy(S_mat.memptr(), S_mat_gpu, S_mat.n_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // return S_mat to its original order.
    S_mat = S_mat(undo_sorted_indices, undo_sorted_indices);
    // S_mat.print("Smat gpu");
    return 0;
}


int eval_Hcoremat_sort_inside(Molecule_basisGPU& system, Molecule_basis& system_cpu, arma::mat &H_mat){
    const size_t nbsf = system_cpu.mAOs.size();
    H_mat.set_size(nbsf,nbsf);

    // Carry out sorting AOs on CPU
    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices, sorted_offs;
    sort_AOs(system_cpu.mAOs, sorted_AOs, sorted_indices, sorted_offs);
    size_t p_start_ind = sorted_offs(1); // The index of the first p AO
    // get the undo sorted indices
    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    
    // Copy Sorted AOs. In hindsight, probably unnecessary and couldve just copied via construct_TV;
    AOGPU* mAO_array_gpu;
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


    arma::mat T_mat(nbsf,nbsf);
    arma::mat V_mat(nbsf,nbsf);
    // Copy T and V matrices
    double *T_mat_gpu, *V_mat_gpu;
    cudaMalloc((void**)&T_mat_gpu, nbsf * nbsf* sizeof(double));
    cudaMemset(T_mat_gpu, 0.0, sizeof(double) * nbsf*nbsf);
    cudaMalloc((void**)&V_mat_gpu, nbsf * nbsf* sizeof(double));
    cudaMemset(V_mat_gpu, 0.0, sizeof(double) * nbsf*nbsf);

    int num_blocks = (nbsf * nbsf /NUM_THREADS)+ 1;
    // Perform construction of H, sorted into blocks of ss, sp, ps,pp
    construct_TV<<<num_blocks,NUM_THREADS>>>(T_mat_gpu, V_mat_gpu, mAO_array_gpu, nbsf, p_start_ind, system.Atom_coords, system.effective_charges, system.num_atom);


    cudaMemcpy(T_mat.memptr(), T_mat_gpu, T_mat.n_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_mat.memptr(), V_mat_gpu, V_mat.n_elem * sizeof(double), cudaMemcpyDeviceToHost);
    // T_mat.print("T_mat");
    // V_mat.print("V_mat");
    
    H_mat = T_mat + V_mat;
    H_mat = H_mat(undo_sorted_indices, undo_sorted_indices);
    // H_mat.print("H_mat");
    return 0;
}


__device__ void construct_S_block(double* Smat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid){

    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    // get the index of the AO in the whole S matrix
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;
    Smat[mu_nu_ind] = eval_Smunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);
}

__global__ void construct_S(double* Smat,  AOGPU* mAOs, size_t nbsf, size_t p_start_ind){
   size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t p_dim = nbsf - p_start_ind; // Now we only have s and p AOs

    construct_S_block(Smat, mAOs, 0, 0, p_start_ind, p_start_ind, nbsf, tid); // ss
    construct_S_block(Smat, mAOs, p_start_ind, 0, p_dim, p_start_ind, nbsf, tid); // ps
    construct_S_block(Smat, mAOs, 0, p_start_ind, p_start_ind, p_dim, nbsf, tid); // sp
    construct_S_block(Smat, mAOs, p_start_ind, p_start_ind, p_dim, p_dim, nbsf, tid); // pp
}



__device__ void construct_T_block(double* Tmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, size_t tid){

    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    // get the index of the AO in the whole T matrix
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;

    // arma::mat Tmat_block(Tmat + mu_start_ind + nu_start_ind*nbsf, num_mu, num_nu, false, true);
    // Tmat_block(mu,nu) = eval_Tmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);
    // int id = mu + mu_start_ind;
    // printf("tid %d computing for mu= %d, nu=%d and mu_nu_ind=%d\n", int(tid), int(mu + mu_start_ind), int(nu + nu_start_ind), int(mu_nu_ind));
    // printf("%d info, R( %1.2f, %1.2f, %1.2f), with angular momentum: %x %x %x, alpha:( %1.2f, %1.2f, %1.2f), dcoef( %1.2f, %1.2f, %1.2f) and len %d\n", int(tid),
    //     mAOs[id].R0[0], mAOs[id].R0[1], mAOs[id].R0[2], mAOs[id].lmn[0], mAOs[id].lmn[1], mAOs[id].lmn[2],
    //     mAOs[id].alpha[0], mAOs[id].alpha[1], mAOs[id].alpha[2], mAOs[id].d_coe[0], mAOs[id].d_coe[1], mAOs[id].d_coe[2], mAOs[id].len);

    Tmat[mu_nu_ind] = eval_Tmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind]);

    // printf("tid %d computes that Tmat[%d]= %1.2f\n", int(tid), int(mu_nu_ind), Tmat[mu_nu_ind]);

}

__device__ void construct_V_block(double* Vmat,  AOGPU* mAOs, size_t mu_start_ind, size_t nu_start_ind, size_t num_mu, size_t num_nu, size_t nbsf, double* Atom_coords, const int* effective_charges, const int num_atom, size_t tid){

    if (tid >= num_mu*num_nu) return;
    // ss
    size_t mu = tid % num_mu;
    size_t nu = tid / num_mu;
    // get the index of the AO in the whole V matrix
    size_t mu_nu_ind = mu + mu_start_ind + (nu + nu_start_ind)*nbsf;
    // printf("tid %d computing for mu= %d, nu=%d and mu_nu_ind=%d\n", int(tid), int(mu), int(nu), int(mu_nu_ind));
    Vmat[mu_nu_ind] = eval_Vmunu(mAOs[mu + mu_start_ind], mAOs[nu + nu_start_ind], Atom_coords, effective_charges, num_atom);
    // printf("tid %d computes that Vmat[%d]= %1.2f\n", int(tid), int(mu_nu_ind),Vmat[mu_nu_ind]);
}


__global__ void construct_TV(double* Tmat, double* Vmat, AOGPU* mAOs, size_t nbsf, size_t p_start_ind, double* Atom_coords, const int* effective_charges,const int num_atom){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t p_dim = nbsf - p_start_ind; // Now we only have s and p AOs
    if (tid==0) printf("construct_TV beginning, p_dim is  %d, nbsf is %d \n", int(p_dim), int(nbsf));

    construct_T_block(Tmat, mAOs, 0,           0,           p_start_ind,  p_start_ind, nbsf, tid); // ss
    construct_T_block(Tmat, mAOs, p_start_ind, 0,           p_dim,        p_start_ind, nbsf, tid); // ps
    construct_T_block(Tmat, mAOs, 0          , p_start_ind, p_start_ind,  p_dim, nbsf, tid); // sp
    construct_T_block(Tmat, mAOs, p_start_ind, p_start_ind, p_dim,        p_dim, nbsf, tid); // pp
    
    construct_V_block(Vmat, mAOs, 0, 0, p_start_ind, p_start_ind, nbsf, Atom_coords, effective_charges, num_atom, tid); // ss
    construct_V_block(Vmat, mAOs, p_start_ind, 0, p_dim, p_start_ind, nbsf, Atom_coords, effective_charges, num_atom, tid); // ps
    construct_V_block(Vmat, mAOs, 0, p_start_ind, p_start_ind, p_dim, nbsf, Atom_coords, effective_charges, num_atom, tid); // sp
    construct_V_block(Vmat, mAOs, p_start_ind, p_start_ind, p_dim, p_dim, nbsf, Atom_coords, effective_charges, num_atom, tid); // pp
    // printf("construct_T is done, tid %d\n", int(tid));

}



__device__ double eval_Smunu(AOGPU &mu, AOGPU &nu){
    
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



__device__ double eval_Vmunu(AOGPU &mu, AOGPU &nu, double* Atom_coords, const int* effective_charges, const int num_atom){
    // nuclear attraction
    
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
    for (size_t c = 0; c < num_atom; c++){
        double* C = Atom_coords + 3*c; // pointer arithmetic to get starting point of C coords
        int Z = effective_charges[c];
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
    assert(n >= 0);
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


__device__ double Combination(int n, int k)
{
  if (n < k || k < 0)
  {
    printf("Comination: the number of elements should be bigger than selection numbers AND two numbers should be positive\n");
    assert(false);
  }
  double result = 1e308;
  if (pow(result, 1.0 / k) < n)
  {
    printf("The Combination number may be bigger than the maxium of double precision\n");
    assert(false);
  }
  double n_d = (double)n;
  result = 1.0;
  int k_small = min(k, n - k);
  for (double j = (double)k_small; j > 0; j--)
  {
    result /= j;
    result *= n_d;
    n_d--;
  }
  return result;
}

__device__ double poly_binom_expans_terms(int n, int ia, int ib, double PminusA_1d, double PminusB_1d){
    // computes the binomial expansion for the terms where ia + ib = n.
    // double total = 0.0;

    double total = 0.0;

    for (int t = 0; t < n + 1; t++){
        if (n - ia <= t && t <= ib){
            total += Combination(ia, n-t) * Combination (ib,t) * pow(PminusA_1d, ia-n+t) * pow(PminusB_1d, ib-t);
        }
    }
    return total;
}

__device__ double overlap_1d(int l1, int l2, double PminusA_1d, double PminusB_1d, double gamma){
    double total  = 0.0;
    for (int i = 0; i < (1+ int(floorf((l1+l2)/2))); i++){
        total += poly_binom_expans_terms(2*i, l1, l2,PminusA_1d, PminusB_1d )* DoubleFactorial(2*i-1)/pow(2*gamma, i);
    }
    return total;
}

__device__ double overlap(double* A,  int l1, int m1, int n1, double alpha, double* B, int l2, int m2, int n2,double beta ){
    double gamma = alpha + beta;
    double P[3];
    double AmBnormSquare = 0.0; // |A-B|^2
    for(int i = 0; i < 3; i++){
        P[i] = (alpha*A[i]+ beta*B[i])/gamma;
        AmBnormSquare += (A[i]-B[i]) * (A[i]-B[i]);
    }
    
    double prefactor = pow(M_PI/gamma,1.5) * exp(-alpha * beta * AmBnormSquare /gamma);
    // printf("Finished prefactor\n");
    // printf("Calculating overlap 1d\n");
    // printf("P( %1.2f, %1.2f, %1.2f),A:( %1.2f, %1.2f, %1.2f), B( %1.2f, %1.2f, %1.2f), alpha (%1.2f), beta (%1.2f)\n", P[0], P[1], P[2],A[0], A[1], A[2],B[0], B[1], B[2], alpha, beta);
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

// Calculate factorial(i)/factorial(r)/factorial(u)/factorial(i-2*r-2*u)
__device__ double A_term_helper(int i, int r, int u){
    assert(i >= 2*r+2*u);
    double result = 1.0;
    double n_d = (double)i;
    for (double j = (double)r; j > 0; j--){
        result *= n_d--;
        result /= j;
    }
    for (double j = (double)u; j > 0; j--){
        result *= n_d--;
        result /= j;
    }
    for (double j = (double)(i-2*r-2*u); j > 0; j--){
        result *= n_d--;
        result /= j;
    }
    for (; n_d > 0; n_d--){
        result *= n_d;
    }
    return result;
}

__device__ double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma){

    return pow(-1,i)*poly_binom_expans_terms(i,l1,l2,PAx,PBx)*\
           pow(-1,u)*pow(CPx,i-2*r-2*u)* pow(0.25/gamma,r+u) *A_term_helper(i,r,u);
}


__device__ void A_tensor(double * A, int Imax, int l1, int l2, double PA, double PB, double CP, double g){
    for (int i = 0; i < Imax; i++){
        A[i] = 0.0;
    }

    for (int i = 0; i < Imax; i++){
        for (int r = 0; r < int(floorf(i/2)+1); r++){
            for (int u = 0; u < int(floorf((i-2*r)/2)+1); u++){
                int  I = i - 2*r - u;
                A[I] = A[I] + A_term(i,r,u,l1,l2,PA,PB,CP,g);
            }
        }
    }

    return;
}

__device__ double nuclear_attraction(double *A,int l1, int m1, int n1,double alpha, double *B, int l2, int m2, int n2,double beta, double *C){
    // Formulation from JPS (21) 11, Nov 1966 by H Taketa et. al
    double gamma = alpha + beta;

    double P[3], dPA[3], dPB[3], dPC[3];
    double rab2 = 0.0, rcp2 = 0.0;
    for(int i = 0; i < 3; i++){
        rab2 += (A[i]-B[i]) * (A[i]-B[i]);
        P[i] = (alpha*A[i]+ beta*B[i])/gamma;
        rcp2 += (C[i]-P[i]) * (C[i]-P[i]);
        dPA[i] = P[i] - A[i];
        dPB[i] = P[i] - B[i];
        dPC[i] = P[i] - C[i];
    }
    
    int Imax = l1+l2+1;
    int Jmax = m1+m2+1;
    int Kmax = n1+n2+1;
    double * Ax =new double[Imax];
    double * Ay =new double[Jmax];
    double * Az =new double[Kmax];
    A_tensor(Ax, Imax, l1,l2,dPA[0],dPB[0],dPC[0],gamma);
    A_tensor(Ay, Jmax, m1,m2,dPA[1],dPB[1],dPC[1],gamma);
    A_tensor(Az, Kmax, n1,n2,dPA[2],dPB[2],dPC[2],gamma);

    double total = 0.0;

    // printf("nuclear_attraction: reached for-loops\n");
    for (int I = 0; I < Imax; I++)
        for (int J = 0; J < Jmax; J++)
            for (int K = 0; K < Kmax; K++)
                total += Ax[I]*Ay[J]*Az[K]*Fgamma(I+J+K,rcp2*gamma);
    delete[] Ax;
    delete[] Ay;
    delete[] Az;
                
    double val= -2*M_PI/gamma*exp(-alpha*beta*rab2/gamma)*total;
    return val;
}

} // namespace hcore_gpu