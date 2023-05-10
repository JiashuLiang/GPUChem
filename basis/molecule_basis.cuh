#ifndef MOLECULE_BASIS_CUH
#define MOLECULE_BASIS_CUH

#define ARMA_ALLOW_FAKE_GCC
#include "molecule_basis.h"


// the GPU version of AO (see AO.h)
typedef struct AOGPU {
    double* R0;
    unsigned int* lmn;
    double* alpha;
    double* d_coe;
    int len;
} AOGPU;

// the GPU version of Molecule_basis (see molecule_basis.h)
// Could be converted from unsorted molecule basis or sorted molecule basis
typedef struct Molecule_basisGPU {
    AOGPU* mAOs;
    double* Atom_coords;
    int* effective_charges; // effective charges of atoms, usually the index of the atom in the periodic table, e.g. 1 for H, 6 for C, 8 for O, but could be other values due to the use of minimum basis
    int m_charge;
    int num_atom, num_ao; // num_ao is the total number of AO, num_atom is the number of atoms
    int num_alpha_ele;
    int num_beta_ele;
} Molecule_basisGPU ;

// void copy_ao_to_gpu(const AO& ao, AOGPU& ao_gpu);

// Copy the unsorted molecule basis to GPU
void copy_molecule_basis_to_gpu(const Molecule_basis& mol_basis, Molecule_basisGPU& mol_basis_gpu);
// Copy the sorted molecule basis to GPU
void copy_sorted_molecule_basis_to_gpu(const Molecule_basis& mol_basis, Molecule_basisGPU& mol_basis_gpu);

// Release the memory of AOGPU
void release_aogpu(AOGPU& ao_gpu);
// Release the memory of Molecule_basisGPU
void release_molecule_basis_gpu(Molecule_basisGPU& mol_basis_gpu);

#endif // MOLECULE_BASIS_CUH