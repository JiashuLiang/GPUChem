#include "JKmat.h"
#include <basis/molecule_basis.h>
#include <armadillo>
#include <filesystem>
#include <cmath>

// calculates (i j | k l), each of those is a CGTO basis function
// sorry using i j k l here instead of mu nu si la
double eval_2eint(arma::mat& rys_root, AO& AO_i, AO& AO_j, AO& AO_k, AO& AO_l); 

// rys roots and weights interpolation from the text file
void rysroot(arma::mat& rys_root, double& X, double& t1, double& t2, double& t3, double& w1, double& w2, double& w3);

// 1-dimension Ix evaluation
double Ix_calc(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al, int& nix, int& njx, int& nkx, int& nlx);
// properly ordered Ix integrals
double Ix_calc_ssss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);
double Ix_calc_psss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);
double Ix_calc_ppss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);
double Ix_calc_psps(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);
double Ix_calc_ppps(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);
double Ix_calc_pppp(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al);





int eval_Gmat_RSCF(Molecule_basis& system, arma::mat &Pa_mat, arma::mat &G_mat){
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = system.mAOs.size();
	arma::mat rys_root;
    std::string aux;
    if(const char* env_p = std::getenv("GPUChem_aux")){
        aux = std::string(env_p);
        if (!std::filesystem::is_directory(aux)) {
            throw std::runtime_error("basis/basis_set.cpp: The directory specified by GPUChem_aux does not exist!");
        }
    }

	rys_root.load(aux + "/rys_root.txt");
	// text file contatins rys root (squared) and their weights from X = 0 to 30 (0.01 increment)

	// checking the basis set to see if there is high angular momentum stuff
	for (int mu = 0; mu < nbasis; mu++) {
		if (arma::accu(system.mAOs[mu].lmn) >= 2){
			std::cout << "higher angular momentum basis detected! Can only do s and p";
			return 1;
		}
	}	
	
	// brute force direct SCF - we won't be saving (mu nu | si la)'s
	// arma::mat G_mat(nbasis, nbasis, arma::fill::zeros); // ??????
	// pragma omp parallel for
	for (int mu = 0; mu < nbasis; mu++){
		AO AO_mu = system.mAOs[mu];
		for (int nu = mu; nu < nbasis; nu++){ // simple symmetry
			// each mu nu can be handled bu one GPU block thread
			AO AO_nu = system.mAOs[nu];
			double Gmunu = 0;
			for (int si = 0; si < nbasis; si++){
				AO AO_si = system.mAOs[si];
				for (int la = 0; la < nbasis; la++){
					AO AO_la = system.mAOs[la];
					Gmunu += (2 * eval_2eint(rys_root, AO_mu, AO_nu, AO_si, AO_la) - eval_2eint(rys_root, AO_mu, AO_la, AO_si, AO_nu)) * Pa_mat(si, la);
					// the paper uses (si la) symmetry I guess for the J matrix, but for simplicity not used here
					// also neither presceening (!) nor load balancing (?) is considered here, which might be bad
				}
			}
			G_mat(mu, nu) = Gmunu;
			G_mat(nu, mu) = Gmunu;
		}
	}


	return 0;
}

double eval_2eint(arma::mat& rys_root, AO& AO_i, AO& AO_j, AO& AO_k, AO& AO_l){
	// some dirty work

	// center coordinates
	double xi = AO_i.R0(0); double xj = AO_j.R0(0); double xk = AO_k.R0(0); double xl = AO_l.R0(0);
	double yi = AO_i.R0(1); double yj = AO_j.R0(1); double yk = AO_k.R0(1); double yl = AO_l.R0(1);
	double zi = AO_i.R0(2); double zj = AO_j.R0(2); double zk = AO_k.R0(2); double zl = AO_l.R0(2);

	// angular momentums
	int nix = AO_i.lmn(0); int njx = AO_j.lmn(0); int nkx = AO_k.lmn(0); int nlx = AO_l.lmn(0);
	int niy = AO_i.lmn(1); int njy = AO_j.lmn(1); int nky = AO_k.lmn(1); int nly = AO_l.lmn(1);
	int niz = AO_i.lmn(2); int njz = AO_j.lmn(2); int nkz = AO_k.lmn(2); int nlz = AO_l.lmn(2);

	// number of contracted orbitals
	int Ni = AO_i.alpha.n_elem; int Nj = AO_j.alpha.n_elem; int Nk = AO_k.alpha.n_elem; int Nl = AO_l.alpha.n_elem;

	double int_val = 0;
	for (size_t i = 0; i < Ni; i++){
		for (size_t j = 0; j < Nj; j++){
			for (size_t k = 0; k < Nk; k++){
				for (size_t l = 0; l < Nl; l++){
					// calculates (i j | k l) where each of those are primitive Gaussians

					double ai = AO_i.alpha(i); double aj = AO_j.alpha(j); double ak = AO_k.alpha(k); double al = AO_l.alpha(l);
	                double A = ai + aj;
	                double B = ak + al;
	                double rho = A*B / (A+B);
	                
	                double xA = (ai*xi + aj*xj) / (ai+aj);
	                double xB = (ak*xk + al*xl) / (ak+al);
	                double Dx = rho *(xA - xB)*(xA - xB);
	                
	                double yA = (ai*yi + aj*yj) / (ai+aj);
	                double yB = (ak*yk + al*yl) / (ak+al);
	                double Dy = rho*(yA - yB)*(yA - yB);
	                
	                double zA = (ai*zi + aj*zj) / (ai+aj);
	                double zB = (ak*zk + al*zl) / (ak+al);
	                double Dz = rho*(zA - zB)*(zA - zB);
	                
	                double X = Dx + Dy + Dz;
	                double t1, t2, t3, w1, w2, w3;
	                rysroot(rys_root, X, t1, t2, t3, w1, w2, w3);
	                
	                double prodcoeff = AO_i.d_coe(i)* AO_j.d_coe(j)* AO_k.d_coe(k)* AO_l.d_coe(l)* 2*sqrt(rho/M_PI);
	                double int1 = w1*prodcoeff*Ix_calc(t1,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t1,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t1,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                double int2 = w2*prodcoeff*Ix_calc(t2,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t2,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t2,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                double int3 = w3*prodcoeff*Ix_calc(t3,xi,xj,xk,xl,ai,aj,ak,al,nix,njx,nkx,nlx)
	                                   		  *Ix_calc(t3,yi,yj,yk,yl,ai,aj,ak,al,niy,njy,nky,nly)
	                                   		  *Ix_calc(t3,zi,zj,zk,zl,ai,aj,ak,al,niz,njz,nkz,nlz);
	                int_val += int1 + int2 + int3;
				}
			}
		}
	}
	return int_val;

}


void rysroot(arma::mat& rys_root, double& X, double& t1, double& t2, double& t3, double& w1, double& w2, double& w3){
	// if X <= 30, read from table (X = 0 case is actually Legendre polynomial, rysroot.m doesn't compute that case)
	// if X > 30, use Hermite polynomial n = 6
	if (X <= 30){
		// read from table
		int flr_index = std::floor(X / 0.01) + 1;
		double flr_weight = (0.01 - (X - (flr_index - 1) * 0.01)) / 0.01;
		int cel_index = flr_index + 1;
		double cel_weight = (0.01 - ((cel_index - 1) * 0.01) - X) / 0.01;

		t1 = rys_root(flr_index, 0)*flr_weight + rys_root(cel_index, 0)*cel_weight;
		t2 = rys_root(flr_index, 1)*flr_weight + rys_root(cel_index, 1)*cel_weight;
		t3 = rys_root(flr_index, 2)*flr_weight + rys_root(cel_index, 2)*cel_weight;
		w1 = rys_root(flr_index, 3)*flr_weight + rys_root(cel_index, 3)*cel_weight;
		w2 = rys_root(flr_index, 4)*flr_weight + rys_root(cel_index, 4)*cel_weight;
		w3 = rys_root(flr_index, 5)*flr_weight + rys_root(cel_index, 5)*cel_weight;
	} else { // X > 30
		t1 = 0.436077412 * 0.436077412 / X;
		t2 = 1.335849074 * 1.335849074 / X;
		t3 = 2.350604974 * 2.350604974 / X;
		w1 = 0.724629595 / std::sqrt(X);
		w2 = 0.157067320 / std::sqrt(X);
		w3 = 0.004530010 / std::sqrt(X);
	}
}


double Ix_calc(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al, int& nix, int& njx, int& nkx, int& nlx){
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


double Ix_calc_ssss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double G00 = M_PI / std::sqrt(A*B);

	double Ix = std::exp(-Gx)*G00;
	return Ix;
}

double Ix_calc_psss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;

	double Ix = std::exp(-Gx)*G10;
	return Ix;
}

double Ix_calc_psps(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double C00p = (xB-xk) + A*(xA-xB)*t2/(A+B);
	double B00 = t2 / (2*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G11 = (B00 + C00*C00p)*G00;

	double Ix = std::exp(-Gx)*G11;
	return Ix;
}

double Ix_calc_ppss(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
	double Gx = ai*aj/(ai+aj)*(xi-xj)*(xi-xj)+ak*al/(ak+al)*(xk-xl)*(xk-xl);

	double C00 = (xA-xi) + B*(xB-xA)*t2/(A+B);
	double B00 = t2 / (2*(A+B));
	double B10 = 1/(2*A) - B*t2/(2*A*(A+B));

	double G00 = M_PI / std::sqrt(A*B);
	double G10 = C00*G00;
	double G20 = (B10 + C00*C00)*G00;

	double Ix = std::exp(-Gx)*(G20+(xi-xj)*G10);
	return Ix;
}

double Ix_calc_ppps(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
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

double Ix_calc_pppp(double& t2, double& xi, double& xj, double& xk, double& xl, double& ai, double& aj, double& ak, double& al){
	double xA = (ai*xi + aj*xj) / (ai+aj);
	double xB = (ak*xk + al*xl) / (ak+al);
	double A = ai + aj;
	double B = ak + al;
	double rho = A*B / (A+B);
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