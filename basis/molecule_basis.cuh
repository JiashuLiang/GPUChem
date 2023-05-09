#ifndef MOLECULE_BASIS_CUH
#define MOLECULE_BASIS_CUH

#define ARMA_ALLOW_FAKE_GCC
#include "molecule_basis.h"


typedef struct AOGPU {
    double* R0;
    unsigned int* lmn;
    double* alpha;
    double* d_coe;
    int len;
} AOGPU;

typedef struct Molecule_basisGPU {
    AOGPU* mAOs;
    double* Atom_coords;
    int* effective_charges;
    int m_charge;
    int num_atom, num_ao;
    int num_alpha_ele;
    int num_beta_ele;
} Molecule_basisGPU ;

// void copy_ao_to_gpu(const AO& ao, AOGPU& ao_gpu);
void copy_molecule_basis_to_gpu(const Molecule_basis& mol_basis, Molecule_basisGPU& mol_basis_gpu);
void release_aogpu(AOGPU& ao_gpu);
void release_molecule_basis_gpu(Molecule_basisGPU& mol_basis_gpu);

void copy_sorted_molecule_basis_to_gpu(const Molecule_basis& mol_basis, Molecule_basisGPU& mol_basis_gpu);

#endif // MOLECULE_BASIS_CUH