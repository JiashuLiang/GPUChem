#include "molecule_basis.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cassert>


void copy_molecule_basis_to_gpu(const Molecule_basis& mol_basis, Molecule_basisGPU& mol_basis_gpu) {
    int num_ao = mol_basis.mAOs.size();
    int num_atoms = mol_basis.m_mol.mAtoms.size();

    // Allocate memory on the GPU for the mAOs array
    cudaMalloc((void**)&mol_basis_gpu.mAOs, num_ao * sizeof(AOGPU));

    // Allocate and copy AO data to the GPU
    for (size_t i = 0; i < num_ao; i++) {
        AOGPU ao_gpu;

        // Set length of AOGPU
        ao_gpu.len = mol_basis.mAOs[i].len;
        
        // Allocate memory on the GPU
        cudaMalloc((void**)&ao_gpu.R0, 3 * sizeof(double));
        cudaMalloc((void**)&ao_gpu.lmn, 3 * sizeof(unsigned int));
        cudaMalloc((void**)&ao_gpu.alpha, ao_gpu.len * sizeof(double));
        cudaMalloc((void**)&ao_gpu.d_coe, ao_gpu.len * sizeof(double));

        // Copy the data from the CPU to the GPU
        cudaMemcpy(ao_gpu.R0, mol_basis.mAOs[i].R0.memptr(), 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.lmn, mol_basis.mAOs[i].lmn.memptr(), 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.alpha, mol_basis.mAOs[i].alpha.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ao_gpu.d_coe, mol_basis.mAOs[i].d_coe.memptr(), ao_gpu.len * sizeof(double), cudaMemcpyHostToDevice);

        // Copy the AOGPU object to the mAOs array on the GPU
        cudaMemcpy(mol_basis_gpu.mAOs + i, &ao_gpu, sizeof(AOGPU), cudaMemcpyHostToDevice);
    }
    
    
    
    // Create temporary arrays to hold the atom coordinates and effective charges
    arma::mat atom_coords(3, num_atoms);
    std::vector<int> effective_charges(num_atoms);
    // Fill the temporary arrays with data from the Molecule_basis object
    for (size_t i = 0; i < num_atoms; i++) {
        const Atom& atom = mol_basis.m_mol.mAtoms[i];
        atom_coords.col(i) = atom.m_coord;
        effective_charges[i] = atom.m_effective_charge;
    }
    
    // Allocate memory on the device for Atom coordinates and effective charges
    cudaMalloc((void**)&(mol_basis_gpu.Atom_coords), num_atoms * 3 * sizeof(double));
    cudaMalloc((void**)&(mol_basis_gpu.effective_charges), num_atoms * sizeof(int));
    
    // Copy the Atom_coords and effective_charges arrays from the host to the GPU
    cudaMemcpy(mol_basis_gpu.Atom_coords, atom_coords.memptr(), 3 * num_atoms * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mol_basis_gpu.effective_charges, effective_charges.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);

    mol_basis_gpu.num_atom = num_atoms;
    mol_basis_gpu.num_ao = num_ao;
    mol_basis_gpu.m_charge = mol_basis.m_mol.m_charge;
    mol_basis_gpu.num_alpha_ele = mol_basis.num_alpha_ele;
    mol_basis_gpu.num_beta_ele = mol_basis.num_beta_ele;
}

void release_aogpu(AOGPU& ao_gpu) {
    if (ao_gpu.R0) {
        cudaFree(ao_gpu.R0);
        ao_gpu.R0 = nullptr;
    }
    if (ao_gpu.lmn) {
        cudaFree(ao_gpu.lmn);
        ao_gpu.lmn = nullptr;
    }
    if (ao_gpu.alpha) {
        cudaFree(ao_gpu.alpha);
        ao_gpu.alpha = nullptr;
    }
    if (ao_gpu.d_coe) {
        cudaFree(ao_gpu.d_coe);
        ao_gpu.d_coe = nullptr;
    }
}

void release_molecule_basis_gpu(Molecule_basisGPU& mol_basis_gpu) {
    if (mol_basis_gpu.mAOs) {
        // Create a temporary AOGPU object to hold data from the GPU
        AOGPU ao_gpu_temp;

        // Free memory for each AOGPU object
        for (int i = 0; i < mol_basis_gpu.num_ao; i++) {
            // Copy the AOGPU object from the GPU to the temporary AOGPU object on the host
            cudaMemcpy(&ao_gpu_temp, mol_basis_gpu.mAOs + i, sizeof(AOGPU), cudaMemcpyDeviceToHost);

            // Release the GPU memory associated with the AOGPU object
            release_aogpu(ao_gpu_temp);
        }
        cudaFree(mol_basis_gpu.mAOs);
        mol_basis_gpu.mAOs = nullptr;
    }

    if (mol_basis_gpu.Atom_coords) {
        cudaFree(mol_basis_gpu.Atom_coords);
        mol_basis_gpu.Atom_coords = nullptr;
    }

    if (mol_basis_gpu.effective_charges) {
        cudaFree(mol_basis_gpu.effective_charges);
        mol_basis_gpu.effective_charges = nullptr;
    }
}
