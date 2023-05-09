#include "JKmat.cuh"
#include <basis/molecule_basis.h>
#include <cmath>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define rys_root(flr_index, x) rys_root[(flr_index) + x * rys_root_dim1]
#define Schwarz_mat(flr_index, x) rys_root[(flr_index) + x * rys_root_dim1]
#define threadsPerBlock_forNbasisSquare 64
#define OneDemension_threadsPerBlock 8


// calculates (i j | k l), each of those is a CGTO basis function
// sorry using i j k l here instead of mu nu si la
__device__ double eval_2eint(double *rys_root, AOGPU& AO_i, AOGPU& AO_j, AOGPU& AO_k, AOGPU& AO_l, int rys_root_dim1); 

// rys roots and weights interpolation from the text file
__device__ void rysroot(double *rys_root, double X, double& t1, double& t2, double& t3, double& w1, double& w2, double& w3, int rys_root_dim1);

// 1-dimension Ix evaluation
__device__ double Ix_calc(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al, int nix, int njx, int nkx, int nlx);
// properly ordered Ix integrals
__device__ double Ix_calc_ssss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);
__device__ double Ix_calc_psss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);
__device__ double Ix_calc_ppss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);
__device__ double Ix_calc_psps(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);
__device__ double Ix_calc_ppps(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);
__device__ double Ix_calc_pppp(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al);




// GPU kernal to calculate G_mat = 2 * J_mat - K_mat as vector operation
__global__ void eval_Gmat_RSCF_kernel(double *J_mat, double *K_mat, double *G_mat, int nbasis){
	int mu = blockIdx.x * blockDim.x + threadIdx.x;
	if (mu < nbasis * nbasis){
		G_mat[mu] = 2 * J_mat[mu] - K_mat[mu];
	}
}
int eval_Gmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double *Pa_mat, double *G_mat, 
					int rys_root_dim1){
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = system.num_ao;

	// to find the maximum element in Schwarz_mat 
	thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(Schwarz_mat);
	thrust::device_ptr<double> max_ptr = thrust::max_element(dev_ptr, dev_ptr + nbasis * nbasis);
	double schwarz_max = *max_ptr;

	// Allocate GPU memory for J and K matrices
	double *J_mat, *K_mat;
	cudaMallocManaged((void **)&J_mat, nbasis * nbasis * sizeof(double));
	cudaMallocManaged((void **)&K_mat, nbasis * nbasis * sizeof(double));
	// Set J and K matrices to zero
	cudaMemset(J_mat, 0, nbasis * nbasis * sizeof(double));
	cudaMemset(K_mat, 0, nbasis * nbasis * sizeof(double));

	// Calculate J_mat and K_mat on GPU using eval_Jmat_RSCF and eval_Kmat_RSCF
	// eval_Jmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, schwarz_max, Pa_mat, J_mat, rys_root_dim1);
	// eval_Kmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, schwarz_max, Pa_mat, K_mat, rys_root_dim1);
	eval_JKmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, schwarz_max, Pa_mat, J_mat, K_mat, rys_root_dim1);


	// Calculate G_mat = 2 * J_mat - K_mat on GPU using eval_Gmat_RSCF_kernel
	int blockSize = 256;
	int numBlocks = (nbasis * nbasis + blockSize - 1) / blockSize;
	eval_Gmat_RSCF_kernel<<<numBlocks, blockSize>>>(J_mat, K_mat, G_mat, nbasis);

	return 0;
}


__global__ void eval_JKmat_RSCF_kernel(AOGPU* d_mAOs, double* d_rys_root, double* d_Schwarz_mat, 
		double schwarz_tol_sq, double schwarz_max, int nbasis, double* d_Pa_mat, double* d_J_mat, double *d_K_mat, int rys_root_dim1) {
    int mu = blockIdx.x;
    int nu = blockIdx.y;

    if (mu >= nbasis || nu >= nbasis || mu > nu) {
        return;
    }

    if (d_Schwarz_mat[nu * nbasis + mu] * schwarz_max < schwarz_tol_sq) {
        return;
    }


	int numblocks_1d = (nbasis + OneDemension_threadsPerBlock - 1) / OneDemension_threadsPerBlock;
    int si_block_idx = blockIdx.z / numblocks_1d;
    int la_block_idx = blockIdx.z % numblocks_1d;

	if (si_block_idx > la_block_idx) {
		return;
	}


    int si = threadIdx.x + blockDim.x * si_block_idx;
    int la = threadIdx.y + blockDim.y * la_block_idx;

    // Use shared memory to store partial sums
    __shared__ double partial_sums[threadsPerBlock_forNbasisSquare];

    // Initialize shared memory
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
    partial_sums[tid] = 0;

    if (si < nbasis && la < nbasis && si <= la) {
        AOGPU AO_mu = d_mAOs[mu];
        AOGPU AO_nu = d_mAOs[nu];
        AOGPU AO_si = d_mAOs[si];
        AOGPU AO_la = d_mAOs[la];
        
		if (d_Schwarz_mat[nu * nbasis + mu] * d_Schwarz_mat[la * nbasis + si] > schwarz_tol_sq) {
            double value = eval_2eint(d_rys_root, AO_mu, AO_nu, AO_si, AO_la, rys_root_dim1);
			// print index, R0, lmn of all AOs and eval_2eint value to debug
			// printf("mu = %d, nu = %d, si = %d, la = %d, value = %f\n", mu, nu, si, la, value);
			// printf("mu index = %d, R0 = (%f, %f, %f), lmn = (%d, %d, %d)\n", mu, AO_mu.R0[0], AO_mu.R0[1], AO_mu.R0[2], AO_mu.lmn[0], AO_mu.lmn[1], AO_mu.lmn[2]);
			// printf("nu index = %d, R0 = (%f, %f, %f), lmn = (%d, %d, %d)\n", nu, AO_nu.R0[0], AO_nu.R0[1], AO_nu.R0[2], AO_nu.lmn[0], AO_nu.lmn[1], AO_nu.lmn[2]);
			// printf("si index = %d, R0 = (%f, %f, %f), lmn = (%d, %d, %d)\n", si, AO_si.R0[0], AO_si.R0[1], AO_si.R0[2], AO_si.lmn[0], AO_si.lmn[1], AO_si.lmn[2]);
			// printf("la index = %d, R0 = (%f, %f, %f), lmn = (%d, %d, %d)\n", la, AO_la.R0[0], AO_la.R0[1], AO_la.R0[2], AO_la.lmn[0], AO_la.lmn[1], AO_la.lmn[2]);

			// for K matrix
            atomicAdd(&d_K_mat[la * nbasis + mu], value * d_Pa_mat[nu * nbasis + si]);
			if(mu != nu)
				atomicAdd(&d_K_mat[la * nbasis + nu], value * d_Pa_mat[mu * nbasis + si]);

			if (si == la){
				partial_sums[tid] = value * d_Pa_mat[la * nbasis + si]; // for J matrix
				// no need to calculate K matrix
			}
			else{
				partial_sums[tid] = 2 * value * d_Pa_mat[la * nbasis + si]; // for J matrix
				// for K matrix
				atomicAdd(&d_K_mat[si * nbasis + mu], value * d_Pa_mat[nu * nbasis + la]);
				if(mu != nu)
					atomicAdd(&d_K_mat[si * nbasis + nu], value * d_Pa_mat[mu * nbasis + la]);
			}
			
        }
    }

    __syncthreads();

    // Perform parallel reduction
    for (int stride = threadsPerBlock_forNbasisSquare / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in the block updates the J_mat with the accumulated value
    if (tid == 0) {
        atomicAdd(&d_J_mat[nu * nbasis + mu], partial_sums[0]);
        if(mu != nu)
            atomicAdd(&d_J_mat[mu * nbasis + nu], partial_sums[0]);
		// print J_mat[nu * nbasis + mu] and J_mat[mu * nbasis + nu] to debug
		// printf("J_mat[%d, %d] = %f, J_mat[%d, %d] = %f\n", nu, mu, d_J_mat[nu * nbasis + mu], mu, nu, d_J_mat[mu * nbasis + nu]);
    }
}



int eval_JKmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
		double *Pa_mat, double *J_mat, double *K_mat, int rys_root_dim1){

	int nbasis = system.num_ao;
    double schwarz_tol_sq = schwarz_tol * schwarz_tol;

    // Define the grid and block dimensions
	dim3 blockDim(OneDemension_threadsPerBlock, OneDemension_threadsPerBlock); // OneDemension_threadsPerBlock^2 threads per block
	int numblocks_1d = (nbasis + OneDemension_threadsPerBlock - 1) / OneDemension_threadsPerBlock;
	dim3 gridDim(nbasis, nbasis, numblocks_1d * numblocks_1d);
    
    // Call the kernel
    eval_JKmat_RSCF_kernel<<<gridDim, blockDim>>>(system.mAOs, rys_root, Schwarz_mat, schwarz_tol_sq, schwarz_max, 
			nbasis, Pa_mat, J_mat, K_mat, rys_root_dim1);

    // Wait for the kernel to finish execution and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in eval_JKmat_RSCF_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__global__ void eval_Jmat_RSCF_kernel(AOGPU* d_mAOs, double* d_rys_root, double* d_Schwarz_mat, 
		double schwarz_tol_sq, double schwarz_max, int nbasis, double* d_Pa_mat, double* d_J_mat, int rys_root_dim1) {
    int mu = blockIdx.x;
    int nu = blockIdx.y;

    if (mu >= nbasis || nu >= nbasis || mu > nu) {
        return;
    }

    if (d_Schwarz_mat[nu * nbasis + mu] * schwarz_max < schwarz_tol_sq) {
        return;
    }

    int index = threadIdx.x + blockDim.x * blockIdx.z;
    int si = index / nbasis;
    int la = index % nbasis;

    // Use shared memory to store partial sums
    __shared__ double partial_sums[threadsPerBlock_forNbasisSquare];

    // Initialize shared memory
    partial_sums[threadIdx.x] = 0;

    if (si < nbasis && la < nbasis) {
        AOGPU AO_mu = d_mAOs[mu];
        AOGPU AO_nu = d_mAOs[nu];
        AOGPU AO_si = d_mAOs[si];
        AOGPU AO_la = d_mAOs[la];
        
		if (d_Schwarz_mat[nu * nbasis + mu] * d_Schwarz_mat[la * nbasis + si] > schwarz_tol_sq) {
            double value = eval_2eint(d_rys_root, AO_mu, AO_nu, AO_si, AO_la, rys_root_dim1) * d_Pa_mat[la * nbasis + si];
            partial_sums[threadIdx.x] = value;
        }
    }

    __syncthreads();

    // Perform parallel reduction
    for (int stride = threadsPerBlock_forNbasisSquare / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The first thread in the block updates the J_mat with the accumulated value
    if (threadIdx.x == 0) {
        atomicAdd(&d_J_mat[nu * nbasis + mu], partial_sums[0]);
        if(mu != nu)
            atomicAdd(&d_J_mat[mu * nbasis + nu], partial_sums[0]);
    }
}

int eval_Jmat_RSCF(Molecule_basisGPU& system, double *rys_root, double *Schwarz_mat, double schwarz_tol, double schwarz_max, 
		double *Pa_mat, double *J_mat, int rys_root_dim1){

	int nbasis = system.num_ao;
    double schwarz_tol_sq = schwarz_tol * schwarz_tol;

    // Define the grid and block dimensions
	int threadsPerBlock = threadsPerBlock_forNbasisSquare; 
	int gridDimZ = (nbasis * nbasis + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blockDim(threadsPerBlock);
	dim3 gridDim(nbasis, nbasis, gridDimZ);
	// dim3 blockDim(1);
	// dim3 gridDim(1, 1, 1);
    
    // Call the kernel
    eval_Jmat_RSCF_kernel<<<gridDim, blockDim>>>(system.mAOs, rys_root, Schwarz_mat, schwarz_tol_sq, schwarz_max, 
			nbasis, Pa_mat, J_mat, rys_root_dim1);

    // Wait for the kernel to finish execution and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in eval_Jmat_RSCF_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__global__ void eval_Kmat_RSCF_kernel(AOGPU* d_mAOs, double* d_rys_root, double* d_Schwarz_mat, double schwarz_tol_sq, int nbasis, double* d_Pa_mat, double* d_K_mat, int rys_root_dim1) {
    int mu = blockIdx.x;
    int nu = blockIdx.y;

    if (mu >= nbasis || nu >= nbasis || mu > nu) {
        return;
    }

    int index = threadIdx.x + blockDim.x * blockIdx.z;
    int si = index % nbasis;
    int la = index / nbasis;

    // Use shared memory to store partial sums
    __shared__ double partial_sums[threadsPerBlock_forNbasisSquare];

    // Initialize shared memory
    partial_sums[threadIdx.x] = 0;

    if (si < nbasis && la < nbasis) {
        AOGPU AO_mu = d_mAOs[mu];
        AOGPU AO_nu = d_mAOs[nu];
        AOGPU AO_si = d_mAOs[si];
        AOGPU AO_la = d_mAOs[la];

        if (d_Schwarz_mat[la * nbasis + mu] * d_Schwarz_mat[nu * nbasis + si] > schwarz_tol_sq) {
            double value = eval_2eint(d_rys_root, AO_mu, AO_la, AO_si, AO_nu, rys_root_dim1) * d_Pa_mat[la * nbasis + si];
            partial_sums[threadIdx.x] = value;
        }
    }

    __syncthreads();

    // Perform parallel reduction
    for (int stride = threadsPerBlock_forNbasisSquare / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The first thread in the block updates the K_mat with the accumulated value
    if (threadIdx.x == 0) {
        atomicAdd(&d_K_mat[nu * nbasis + mu], partial_sums[0]);
        if(mu != nu)
            atomicAdd(&d_K_mat[mu * nbasis + nu], partial_sums[0]);
    }
}

int eval_Kmat_RSCF(Molecule_basisGPU& system, double* rys_root, double* Schwarz_mat, double schwarz_tol, double schwarz_max, double* Pa_mat, double* K_mat, int rys_root_dim1) {
    int nbasis = system.num_ao;
    double schwarz_tol_sq = schwarz_tol * schwarz_tol;

    // Define the grid and block dimensions
	int threadsPerBlock = threadsPerBlock_forNbasisSquare;
	int gridDimZ = (nbasis * nbasis + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blockDim(threadsPerBlock);
	dim3 gridDim(nbasis, nbasis, gridDimZ);

    // Call the kernel
    eval_Kmat_RSCF_kernel<<<gridDim, blockDim>>>(system.mAOs, rys_root, Schwarz_mat, schwarz_tol_sq, nbasis, Pa_mat, K_mat, rys_root_dim1);

    // Wait for the kernel to finish execution and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in eval_Kmat_RSCF_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


// the GPU kernal for eval_Schwarzmat
__global__ void eval_Schwarzmat_GPU(AOGPU * mAOs, double *rys_root, double *Schwarz_mat, int nbasis, int rys_root_dim1){
	int mu = blockIdx.x * blockDim.x + threadIdx.x;
	int nu = blockIdx.y * blockDim.y + threadIdx.y;
	if (mu >= nbasis || nu >= nbasis || mu > nu)
		return;
	// each thread handles one (mu nu) pair
	AOGPU AO_mu = mAOs[mu];
	AOGPU AO_nu = mAOs[nu];
	double Schmunu = eval_2eint(rys_root, AO_mu, AO_nu, AO_mu, AO_nu, rys_root_dim1);
	Schwarz_mat[mu * nbasis + nu] = Schmunu;
	Schwarz_mat[nu * nbasis + mu] = Schmunu;
}





__device__ double eval_2eint(double *rys_root, AOGPU& AO_i, AOGPU& AO_j, AOGPU& AO_k, AOGPU& AO_l, int rys_root_dim1){
	// some dirty work

	// center coordinates
	double xi = AO_i.R0[0]; double xj = AO_j.R0[0]; double xk = AO_k.R0[0]; double xl = AO_l.R0[0];
	double yi = AO_i.R0[1]; double yj = AO_j.R0[1]; double yk = AO_k.R0[1]; double yl = AO_l.R0[1];
	double zi = AO_i.R0[2]; double zj = AO_j.R0[2]; double zk = AO_k.R0[2]; double zl = AO_l.R0[2];

	// angular momentums
	int nix = AO_i.lmn[0]; int njx = AO_j.lmn[0]; int nkx = AO_k.lmn[0]; int nlx = AO_l.lmn[0];
	int niy = AO_i.lmn[1]; int njy = AO_j.lmn[1]; int nky = AO_k.lmn[1]; int nly = AO_l.lmn[1];
	int niz = AO_i.lmn[2]; int njz = AO_j.lmn[2]; int nkz = AO_k.lmn[2]; int nlz = AO_l.lmn[2];

	// number of contracted orbitals
	int Ni = AO_i.len; int Nj = AO_j.len; int Nk = AO_k.len; int Nl = AO_l.len;

	double int_val = 0;
	for (int i = 0; i < Ni; i++){
		for (int j = 0; j < Nj; j++){
			for (int k = 0; k < Nk; k++){
				for (int l = 0; l < Nl; l++){
					// calculates (i j | k l) where each of those are primitive Gaussians

					double ai = AO_i.alpha[i]; double aj = AO_j.alpha[j]; double ak = AO_k.alpha[k]; double al = AO_l.alpha[l];
	                double A = ai + aj;
	                double B = ak + al;
	                double rho = A*B / (A+B);
	                
	                double xA = (ai*xi + aj*xj) / (ai+aj);
	                double xB = (ak*xk + al*xl) / (ak+al);
	                double Dx = rho * (xA - xB) * (xA - xB);
	                
	                double yA = (ai*yi + aj*yj) / (ai+aj);
	                double yB = (ak*yk + al*yl) / (ak+al);
	                double Dy = rho * (yA - yB) * (yA - yB);
	                
	                double zA = (ai*zi + aj*zj) / (ai+aj);
	                double zB = (ak*zk + al*zl) / (ak+al);
	                double Dz = rho * (zA - zB) * (zA - zB);
	                
	                double X = Dx + Dy + Dz;
	                double t1, t2, t3, w1, w2, w3;
	                rysroot(rys_root, X, t1, t2, t3, w1, w2, w3, rys_root_dim1);
	                
	                double prodcoeff = AO_i.d_coe[i]* AO_j.d_coe[j]* AO_k.d_coe[k]* AO_l.d_coe[l]* 2*sqrt(rho/M_PI);
	                double integral1 = w1*prodcoeff*Ix_calc(t1,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t1,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t1,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                double integral2 = w2*prodcoeff*Ix_calc(t2,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t2,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t2,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                double integral3 = w3*prodcoeff*Ix_calc(t3,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t3,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t3,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                int_val += integral1 + integral2 + integral3;
					// // print all to debug
					// printf("i: %d, j: %d, k: %d, l: %d, int1: %f, int2: %f, int3: %f\n", i, j, k, l, integral1, integral2, integral3);
					// print t1,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx
					// printf("t1: %f, xi: %f, xj: %f, xk: %f, xl: %f, ai: %f, aj: %f, ak: %f, al: %f, nix: %d, njx: %d, nkx: %d, nlx: %d\n", t1, xi, xj, xk, xl, ai, aj, ak, al, nix, njx, nkx, nlx);
					
				}
			}
		}
	}
	return int_val;

}

__device__ double lagrange_interpolate(double X, double flr, double mid, double cel){
	// 3 point lagrange interpolation -- flr .. X .. mid .. cel
	double flr_X = 0.01 * std::floor(X / 0.01);
	double mid_X = flr_X + 0.01;
	double cel_X = flr_X + 0.02;

	return flr*(X-mid_X)*(X-cel_X)/0.0002 - mid*(X-flr_X)*(X-cel_X)/0.0001 + cel*(X-mid_X)*(X-flr_X)/0.0002;
}

__device__ void rysroot(double *rys_root, double X, double& t1, double& t2, double& t3, double& w1, double& w2, double& w3, int rys_root_dim1){
	// if X <= 30, read from table (X = 0 case is actually Legendre polynomial, rysroot.m doesn't compute that case)
	// if X > 30, use Hermite polynomial n = 6
	if (X <= 29.99){
		// read from table
		int flr_index = std::floor(X / 0.01);
		int mid_index = flr_index + 1;
		int cel_index = flr_index + 2;

		t1 = lagrange_interpolate(X, rys_root(flr_index, 0), rys_root(mid_index, 0), rys_root(cel_index, 0));
		t2 = lagrange_interpolate(X, rys_root(flr_index, 1), rys_root(mid_index, 1), rys_root(cel_index, 1));
		t3 = lagrange_interpolate(X, rys_root(flr_index, 2), rys_root(mid_index, 2), rys_root(cel_index, 2));
		w1 = lagrange_interpolate(X, rys_root(flr_index, 3), rys_root(mid_index, 3), rys_root(cel_index, 3));
		w2 = lagrange_interpolate(X, rys_root(flr_index, 4), rys_root(mid_index, 4), rys_root(cel_index, 4));
		w3 = lagrange_interpolate(X, rys_root(flr_index, 5), rys_root(mid_index, 5), rys_root(cel_index, 5));

		t1 = t1*t1; t2 = t2*t2; t3 = t3*t3;
		// std::cout << "flr_index, X, t1, w1 " << flr_index << " " << X << " " << t1 << " " << w1 << " " << std::endl;
	} else { // X > 30
		t1 = 0.436077412 * 0.436077412 / X;
		t2 = 1.335849074 * 1.335849074 / X;
		t3 = 2.350604974 * 2.350604974 / X;
		w1 = 0.724629595 / std::sqrt(X);
		w2 = 0.157067320 / std::sqrt(X);
		w3 = 0.004530010 / std::sqrt(X);
	}
}


__device__ double Ix_calc(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al, int nix, int njx, int nkx, int nlx){
	// compute Ix(t2) values
	// arrange ijkl order to properly ordered integrals and call Ix_calc_(ordered)

	int nx = nix + njx + nkx + nlx;
	// ssss type 
	if (nx == 0)
		return Ix_calc_ssss(t2,xi,xj,xk,xl,ai,aj,ak,al);

	// pppp type
	if (nx == 4)
		return Ix_calc_pppp(t2,xi,xj,xk,xl,ai,aj,ak,al);

	// psss type
	if (nx == 1){
		if (nix == 1)
			return Ix_calc_psss(t2,xi,xj,xk,xl,ai,aj,ak,al);
		else if (njx == 1)
			return Ix_calc_psss(t2,xj,xi,xk,xl,aj,ai,ak,al);
		else if (nkx == 1)
			return Ix_calc_psss(t2,xk,xl,xi,xj,ak,al,ai,aj);
		else // (nlx == 1)
			return Ix_calc_psss(t2,xl,xk,xi,xj,al,ak,ai,aj);
	}

	// ppps type
	if (nx == 3){
		if (nlx == 0)
			return Ix_calc_ppps(t2,xi,xj,xk,xl,ai,aj,ak,al);
		else if (nkx == 0)
			return Ix_calc_ppps(t2,xi,xj,xl,xk,ai,aj,al,ak);
		else if (njx == 0)
			return Ix_calc_ppps(t2,xk,xl,xi,xj,ak,al,ai,aj);
		else // (nix == 0)
			return Ix_calc_ppps(t2,xk,xl,xj,xi,ak,al,aj,ai);
	}

	// ppss type and psps type
	if (nx == 2){
		int nx1 = nix + njx;
		if (nx1 == 2)
			return Ix_calc_ppss(t2,xi,xj,xk,xl,ai,aj,ak,al);
		else if (nx1 == 0)
			return Ix_calc_ppss(t2,xk,xl,xi,xj,ak,al,ai,aj);
		else { // (nx1 == 1)
			if ((nix == 1) && (nkx == 1))
				return Ix_calc_psps(t2,xi,xj,xk,xl,ai,aj,ak,al);
			else if ((nix == 1) && (nlx == 1))
				return Ix_calc_psps(t2,xi,xj,xl,xk,ai,aj,al,ak);
			else if ((njx == 1) && (nkx == 1))
				return Ix_calc_psps(t2,xj,xi,xk,xl,aj,ai,ak,al);
			else // ((njx == 1) && (nlx == 1))
				return Ix_calc_psps(t2,xj,xi,xl,xk,aj,ai,al,ak);
		}
	}
	return 0.0;
}


__device__ double Ix_calc_ssss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double G00 = M_PI / std::sqrt(A*B);

	double Ix = std::exp(-Gx)*G00;
	return Ix;
}

__device__ double Ix_calc_psss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;

	double Ix = std::exp(-Gx)*G10;
	return Ix;
}

__device__ double Ix_calc_psps(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double C00p = (xB-xk) + A*(xA-xB)*t2/(A+B);
	double B00 = t2 / (2*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G11 = (B00 + C00*C00p)*G00;

	double Ix = std::exp(-Gx)*G11;
	return Ix;
}

__device__ double Ix_calc_ppss(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double B10 = 1/(2*A) - B*t2/(2*A*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;
	double G20 = (B10 + C00*C00)*G00;

	double Ix = std::exp(-Gx)*(G20+(xi-xj)*G10);
	return Ix;
}

__device__ double Ix_calc_ppps(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double C00p = (xB-xk) + A*(xA-xB)*t2/(A+B);
	double B00 = t2 / (2*(A+B));
	double B10 = 1/(2*A) - B*t2/(2*A*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;
	double G11 = (B00 + C00*C00p)*G00;
	double G20 = (B10 + C00*C00)*G00;
	double G21 = 2*B00*G10 + C00p*G20;

	double Ix = std::exp(-Gx)*(G21+(xi-xj)*G11);
	return Ix;
}

__device__ double Ix_calc_pppp(double t2, double xi, double xj, double xk, double xl, double ai, double aj, double ak, double al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double C00p = (xB-xk) + A*(xA-xB)*t2/(A+B);
	double B00 = t2 / (2*(A+B));
	double B10 = 1/(2*A) - B*t2/(2*A*(A+B));
	double B01p = 1/(2*B) - A*t2/(2*B*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;
	double G01 = C00p*G00;
	double G11 = (B00 + C00*C00p)*G00;
	double G20 = (B10 + C00*C00)*G00;
	double G02 = (B01p + C00p*C00p)*G00;
	double G21 = 2*B00*G10 + C00p*G20;
	double G12 = 2*B00*G01 + C00*G02;
	double G22 = B01p*G20 + 2*B00*G11 + C00p*G21;

	double Ix = std::exp(-Gx)*(G22+(xi-xj)*G12+(xk-xl)*G21+(xi-xj)*(xk-xl)*G11);
	return Ix;
}
