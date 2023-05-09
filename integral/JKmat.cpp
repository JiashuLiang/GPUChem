#include "JKmat.h"
#include <basis/molecule_basis.h>
#include <armadillo>
#include <cmath>

// calculates (i j | k l), each of those is a CGTO basis function
// sorry using i j k l here instead of mu nu si la
double eval_2eint(arma::mat &rys_root, AO &AO_i, AO &AO_j, AO &AO_k, AO &AO_l);

// rys roots and weights interpolation from the text file
void rysroot(arma::mat &rys_root, double &X, double &t1, double &t2, double &t3, double &w1, double &w2, double &w3);

// 1-dimension Ix evaluation
double Ix_calc(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al, int &nix, int &njx, int &nkx, int &nlx);
// properly ordered Ix integrals
double Ix_calc_ssss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);
double Ix_calc_psss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);
double Ix_calc_ppss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);
double Ix_calc_psps(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);
double Ix_calc_ppps(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);
double Ix_calc_pppp(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al);

int eval_Gmat_RSCF(Molecule_basis &system, arma::mat &rys_root, arma::mat &Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &G_mat)
{
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = system.mAOs.size();

	// checking the basis set to see if there is high angular momentum stuff
	for (int mu = 0; mu < nbasis; mu++)
	{
		if (arma::accu(system.mAOs[mu].lmn) >= 2)
		{
			std::cout << "higher angular momentum basis detected! Can only do s and p";
			return 1;
		}
	}

	arma::mat J_mat(nbasis, nbasis);
	arma::mat K_mat(nbasis, nbasis);
	eval_Jmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, Pa_mat, J_mat);
	eval_Kmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, Pa_mat, K_mat);
	G_mat = 2 * J_mat - K_mat;

	// double schwarz_tol_sq = schwarz_tol * schwarz_tol;

	// // brute force direct SCF - we won't be saving (mu nu | si la)'s
	// // # pragma omp parallel for
	// for (int mu = 0; mu < nbasis; mu++){
	// 	AO AO_mu = system.mAOs[mu];
	// 	for (int nu = mu; nu < nbasis; nu++){ // simple symmetry
	// 		// each mu nu can be handled bu one GPU block thread
	// 		// if (Schwarz_mat(mu, nu) < schwarz_tol_sq)
	// 		// 	continue;
	// 		AO AO_nu = system.mAOs[nu];
	// 		double Gmunu = 0;
	// 		for (int si = 0; si < nbasis; si++){
	// 			AO AO_si = system.mAOs[si];
	// 			for (int la = 0; la < nbasis; la++){
	// 				AO AO_la = system.mAOs[la];
	// 				double munusila = 0;
	// 				double mulasinu = 0;
	// 				if (Schwarz_mat(mu, nu) * Schwarz_mat(si, la) < schwarz_tol_sq)
	// 					munusila = 0;
	// 				else
	// 					munusila = eval_2eint(rys_root, AO_mu, AO_nu, AO_si, AO_la);
	// 				if (Schwarz_mat(mu, la) * Schwarz_mat(si, nu) < schwarz_tol_sq)
	// 					mulasinu = 0;
	// 				else
	// 					mulasinu = eval_2eint(rys_root, AO_mu, AO_la, AO_si, AO_nu);
	// 				Gmunu += (2 * munusila - mulasinu) * Pa_mat(si, la);
	// 				// the paper uses (si la) symmetry I guess for the J matrix, but for simplicity not used here
	// 				// also neither presceening (!) nor load balancing (?) is considered here, which might be bad
	// 			}
	// 		}
	// 		G_mat(mu, nu) = Gmunu;
	// 		G_mat(nu, mu) = Gmunu;
	// 	}
	// }

	return 0;
}


int eval_Gmat_RSCF(std::vector<AO> &mAOs, arma::mat &rys_root, arma::mat &Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &G_mat)
{
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = mAOs.size();

	// checking the basis set to see if there is high angular momentum stuff
	for (int mu = 0; mu < nbasis; mu++)
	{
		if (arma::accu(mAOs[mu].lmn) >= 2)
		{
			std::cout << "higher angular momentum basis detected! Can only do s and p";
			return 1;
		}
	}

	arma::mat J_mat(nbasis, nbasis);
	arma::mat K_mat(nbasis, nbasis);
	// eval_Jmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, Pa_mat, J_mat);
	// eval_Kmat_RSCF(system, rys_root, Schwarz_mat, schwarz_tol, Pa_mat, K_mat);
	J_mat.zeros();
	K_mat.zeros();
	eval_JKmat_RSCF(mAOs, rys_root, Schwarz_mat, schwarz_tol, Pa_mat, J_mat, K_mat);
	G_mat = 2 * J_mat - K_mat;

	return 0;
}

int eval_JKmat_RSCF(std::vector<AO> &mAOs, arma::mat &rys_root, arma::mat &Schwarz_mat, double schwarz_tol,
					arma::mat &Pa_mat, arma::mat &J_mat, arma::mat &K_mat)
{
	// J_{mu nu} = \sum_{si,la}(mu nu | si la) P_{si la}
	// K_{mu la} = \sum_{si,nu}(mu nu | si la) P_{si nu}
	int nbasis = mAOs.size();
	double schwarz_tol_sq = schwarz_tol * schwarz_tol;
	double schwarz_max = Schwarz_mat.max();

	// direct SCF - we won't be saving (mu nu | si la)'s
	for (int mu = 0; mu < nbasis; mu++)
	{
		AO AO_mu = mAOs[mu];
		for (int nu = mu; nu < nbasis; nu++)
		{ // simple symmetry
			// each mu nu can be handled bu one GPU block thread
			if (Schwarz_mat(mu, nu) * schwarz_max < schwarz_tol_sq)
				continue;
			AO AO_nu = mAOs[nu];
			double Jmunu = 0;
			for (int si = 0; si < nbasis; si++)
			{
				AO AO_si = mAOs[si];
				for (int la = si; la < nbasis; la++)
				{
					AO AO_la = mAOs[la];
					if (Schwarz_mat(mu, nu) * Schwarz_mat(si, la) > schwarz_tol_sq)
					{
						double munusila = eval_2eint(rys_root, AO_mu, AO_nu, AO_si, AO_la);
						// for J matrix
						if (si == la)
							Jmunu += munusila * Pa_mat(si, la);
						else
							Jmunu += 2 * munusila * Pa_mat(si, la);
							
						// for K matrix
						K_mat(mu, la) += munusila * Pa_mat(si, nu);
						if(mu != nu)
							K_mat(nu, la) += munusila * Pa_mat(si, mu);
						if(si != la)
							K_mat(mu, si) += munusila * Pa_mat(la, nu);
						if(mu != nu && si != la)
							K_mat(nu, si) += munusila * Pa_mat(la, mu);
					}
				}
			}
			J_mat(mu, nu) = Jmunu;
			J_mat(nu, mu) = Jmunu;
		}
	}
	
	return 0;
}

int eval_Jmat_RSCF(Molecule_basis &system, arma::mat &rys_root, arma::mat &Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &J_mat)
{
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = system.mAOs.size();
	// arma::mat rys_root;

	double schwarz_tol_sq = schwarz_tol * schwarz_tol;
	double schwarz_max = Schwarz_mat.max();

	// direct SCF - we won't be saving (mu nu | si la)'s
	// pragma omp parallel for
	for (int mu = 0; mu < nbasis; mu++)
	{
		AO AO_mu = system.mAOs[mu];
		for (int nu = mu; nu < nbasis; nu++)
		{ // simple symmetry
			// each mu nu can be handled bu one GPU block thread
			if (Schwarz_mat(mu, nu) * schwarz_max < schwarz_tol_sq)
				continue;
			AO AO_nu = system.mAOs[nu];
			double Jmunu = 0;
			for (int si = 0; si < nbasis; si++)
			{
				AO AO_si = system.mAOs[si];
				if (Schwarz_mat(mu, nu) * Schwarz_mat(si, si) > schwarz_tol_sq)
					Jmunu += eval_2eint(rys_root, AO_mu, AO_nu, AO_si, AO_si) * Pa_mat(si, si);
				for (int la = si + 1; la < nbasis; la++)
				{
					AO AO_la = system.mAOs[la];
					if (Schwarz_mat(mu, nu) * Schwarz_mat(si, la) > schwarz_tol_sq)
						Jmunu += 2 * eval_2eint(rys_root, AO_mu, AO_nu, AO_si, AO_la) * Pa_mat(si, la);
				}
			}
			J_mat(mu, nu) = Jmunu;
			J_mat(nu, mu) = Jmunu;
		}
	}

	return 0;
}

int eval_Kmat_RSCF(Molecule_basis &system, arma::mat &rys_root, arma::mat &Schwarz_mat, double schwarz_tol, arma::mat &Pa_mat, arma::mat &K_mat)
{
	// F = H + G, G is the two-electron part of the Fock matrix
	// G_{mu nu} = \sum_{si,la}[2(mu nu | si la) - (mu la | si nu)] P_{si la}

	int nbasis = system.mAOs.size();
	// arma::mat rys_root;

	double schwarz_tol_sq = schwarz_tol * schwarz_tol;
	double schwarz_max = Schwarz_mat.max();

	// direct SCF - we won't be saving (mu nu | si la)'s
	// pragma omp parallel for
	for (int mu = 0; mu < nbasis; mu++)
	{
		AO AO_mu = system.mAOs[mu];
		for (int nu = mu; nu < nbasis; nu++)
		{ // simple symmetry
			// each mu nu can be handled bu one GPU block thread
			AO AO_nu = system.mAOs[nu];
			double Kmunu = 0;
			for (int si = 0; si < nbasis; si++)
			{
				AO AO_si = system.mAOs[si];
				for (int la = 0; la < nbasis; la++)
				{
					AO AO_la = system.mAOs[la];
					if (Schwarz_mat(mu, la) * Schwarz_mat(si, nu) > schwarz_tol_sq)
						Kmunu += eval_2eint(rys_root, AO_mu, AO_la, AO_si, AO_nu) * Pa_mat(si, la);
				}
			}
			K_mat(mu, nu) = Kmunu;
			K_mat(nu, mu) = Kmunu;
		}
	}

	return 0;
}

int eval_Schwarzmat(Molecule_basis &system, arma::mat &rys_root, arma::mat &Schwarz_mat)
{
	// evaluate (mu nu | mu nu) for Schwarz prescreening
	int nbasis = system.mAOs.size();

	for (size_t mu = 0; mu < nbasis; mu++)
	{
		AO AO_mu = system.mAOs[mu];
		for (size_t nu = mu; nu < nbasis; nu++)
		{
			AO AO_nu = system.mAOs[nu];
			double Schmunu = eval_2eint(rys_root, AO_mu, AO_nu, AO_mu, AO_nu);
			Schwarz_mat(mu, nu) = Schmunu;
			Schwarz_mat(nu, mu) = Schmunu;
		}
	}

	return 0;
}

double eval_2eint(arma::mat &rys_root, AO &AO_i, AO &AO_j, AO &AO_k, AO &AO_l)
{
	// some dirty work

	// center coordinates
	double xi = AO_i.R0(0);
	double xj = AO_j.R0(0);
	double xk = AO_k.R0(0);
	double xl = AO_l.R0(0);
	double yi = AO_i.R0(1);
	double yj = AO_j.R0(1);
	double yk = AO_k.R0(1);
	double yl = AO_l.R0(1);
	double zi = AO_i.R0(2);
	double zj = AO_j.R0(2);
	double zk = AO_k.R0(2);
	double zl = AO_l.R0(2);

	// angular momentums
	int nix = AO_i.lmn(0);
	int njx = AO_j.lmn(0);
	int nkx = AO_k.lmn(0);
	int nlx = AO_l.lmn(0);
	int niy = AO_i.lmn(1);
	int njy = AO_j.lmn(1);
	int nky = AO_k.lmn(1);
	int nly = AO_l.lmn(1);
	int niz = AO_i.lmn(2);
	int njz = AO_j.lmn(2);
	int nkz = AO_k.lmn(2);
	int nlz = AO_l.lmn(2);

	// number of contracted orbitals
	int Ni = AO_i.alpha.n_elem;
	int Nj = AO_j.alpha.n_elem;
	int Nk = AO_k.alpha.n_elem;
	int Nl = AO_l.alpha.n_elem;

	double int_val = 0;
	for (size_t i = 0; i < Ni; i++)
	{
		for (size_t j = 0; j < Nj; j++)
		{
			for (size_t k = 0; k < Nk; k++)
			{
				for (size_t l = 0; l < Nl; l++)
				{
					// calculates (i j | k l) where each of those are primitive Gaussians

					double ai = AO_i.alpha(i);
					double aj = AO_j.alpha(j);
					double ak = AO_k.alpha(k);
					double al = AO_l.alpha(l);
					double A = ai + aj;
					double B = ak + al;
					double rho = A * B / (A + B);

					double xA = (ai * xi + aj * xj) / (ai + aj);
					double xB = (ak * xk + al * xl) / (ak + al);
					double Dx = rho * (xA - xB) * (xA - xB);

					double yA = (ai * yi + aj * yj) / (ai + aj);
					double yB = (ak * yk + al * yl) / (ak + al);
					double Dy = rho * (yA - yB) * (yA - yB);

					double zA = (ai * zi + aj * zj) / (ai + aj);
					double zB = (ak * zk + al * zl) / (ak + al);
					double Dz = rho * (zA - zB) * (zA - zB);

					double X = Dx + Dy + Dz;
					double t1, t2, t3, w1, w2, w3;
					rysroot(rys_root, X, t1, t2, t3, w1, w2, w3);

					double prodcoeff = AO_i.d_coe(i) * AO_j.d_coe(j) * AO_k.d_coe(k) * AO_l.d_coe(l) * 2 * sqrt(rho / M_PI);
					double int1 = w1 * prodcoeff * Ix_calc(t1, xi, xj, xk, xl, ai, aj, ak, al, nix, njx, nkx, nlx) * Ix_calc(t1, yi, yj, yk, yl, ai, aj, ak, al, niy, njy, nky, nly) * Ix_calc(t1, zi, zj, zk, zl, ai, aj, ak, al, niz, njz, nkz, nlz);
					double int2 = w2 * prodcoeff * Ix_calc(t2, xi, xj, xk, xl, ai, aj, ak, al, nix, njx, nkx, nlx) * Ix_calc(t2, yi, yj, yk, yl, ai, aj, ak, al, niy, njy, nky, nly) * Ix_calc(t2, zi, zj, zk, zl, ai, aj, ak, al, niz, njz, nkz, nlz);
					double int3 = w3 * prodcoeff * Ix_calc(t3, xi, xj, xk, xl, ai, aj, ak, al, nix, njx, nkx, nlx) * Ix_calc(t3, yi, yj, yk, yl, ai, aj, ak, al, niy, njy, nky, nly) * Ix_calc(t3, zi, zj, zk, zl, ai, aj, ak, al, niz, njz, nkz, nlz);
					int_val += int1 + int2 + int3;
				}
			}
		}
	}
	return int_val;
}

double lagrange_interpolate(double &X, double &flr, double &mid, double &cel)
{
	// 3 point lagrange interpolation -- flr .. X .. mid .. cel
	double flr_X = 0.01 * std::floor(X / 0.01);
	double mid_X = flr_X + 0.01;
	double cel_X = flr_X + 0.02;

	return flr * (X - mid_X) * (X - cel_X) / 0.0002 - mid * (X - flr_X) * (X - cel_X) / 0.0001 + cel * (X - mid_X) * (X - flr_X) / 0.0002;
}

void rysroot(arma::mat &rys_root, double &X, double &t1, double &t2, double &t3, double &w1, double &w2, double &w3)
{
	// if X <= 30, read from table (X = 0 case is actually Legendre polynomial, rysroot.m doesn't compute that case)
	// if X > 30, use Hermite polynomial n = 6
	if (X <= 29.99)
	{
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

		t1 = t1 * t1;
		t2 = t2 * t2;
		t3 = t3 * t3;
		// std::cout << "flr_index, X, t1, w1 " << flr_index << " " << X << " " << t1 << " " << w1 << " " << std::endl;
	}
	else
	{ // X > 30
		t1 = 0.436077412 * 0.436077412 / X;
		t2 = 1.335849074 * 1.335849074 / X;
		t3 = 2.350604974 * 2.350604974 / X;
		w1 = 0.724629595 / std::sqrt(X);
		w2 = 0.157067320 / std::sqrt(X);
		w3 = 0.004530010 / std::sqrt(X);
	}
}

double Ix_calc(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al, int &nix, int &njx, int &nkx, int &nlx)
{
	// compute Ix(t2) values
	// arrange ijkl order to properly ordered integrals and call Ix_calc_(ordered)

	int nx = nix + njx + nkx + nlx;
	// ssss type
	if (nx == 0)
		return Ix_calc_ssss(t2, xi, xj, xk, xl, ai, aj, ak, al);

	// pppp type
	if (nx == 4)
		return Ix_calc_pppp(t2, xi, xj, xk, xl, ai, aj, ak, al);

	// psss type
	if (nx == 1)
	{
		if (nix == 1)
			return Ix_calc_psss(t2, xi, xj, xk, xl, ai, aj, ak, al);
		else if (njx == 1)
			return Ix_calc_psss(t2, xj, xi, xk, xl, aj, ai, ak, al);
		else if (nkx == 1)
			return Ix_calc_psss(t2, xk, xl, xi, xj, ak, al, ai, aj);
		else // (nlx == 1)
			return Ix_calc_psss(t2, xl, xk, xi, xj, al, ak, ai, aj);
	}

	// ppps type
	if (nx == 3)
	{
		if (nlx == 0)
			return Ix_calc_ppps(t2, xi, xj, xk, xl, ai, aj, ak, al);
		else if (nkx == 0)
			return Ix_calc_ppps(t2, xi, xj, xl, xk, ai, aj, al, ak);
		else if (njx == 0)
			return Ix_calc_ppps(t2, xk, xl, xi, xj, ak, al, ai, aj);
		else // (nix == 0)
			return Ix_calc_ppps(t2, xk, xl, xj, xi, ak, al, aj, ai);
	}

	// ppss type and psps type
	if (nx == 2)
	{
		int nx1 = nix + njx;
		if (nx1 == 2)
			return Ix_calc_ppss(t2, xi, xj, xk, xl, ai, aj, ak, al);
		else if (nx1 == 0)
			return Ix_calc_ppss(t2, xk, xl, xi, xj, ak, al, ai, aj);
		else
		{ // (nx1 == 1)
			if ((nix == 1) && (nkx == 1))
				return Ix_calc_psps(t2, xi, xj, xk, xl, ai, aj, ak, al);
			else if ((nix == 1) && (nlx == 1))
				return Ix_calc_psps(t2, xi, xj, xl, xk, ai, aj, al, ak);
			else if ((njx == 1) && (nkx == 1))
				return Ix_calc_psps(t2, xj, xi, xk, xl, aj, ai, ak, al);
			else // ((njx == 1) && (nlx == 1))
				return Ix_calc_psps(t2, xj, xi, xl, xk, aj, ai, al, ak);
		}
	}
	return 0.0;
}

double Ix_calc_ssss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double G00 = M_PI / std::sqrt(A * B);

	double Ix = std::exp(-Gx) * G00;
	return Ix;
}

double Ix_calc_psss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double xA = (ai * xi + aj * xj) / (ai + aj);
	double xB = (ak * xk + al * xl) / (ak + al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double C00 = (xA - xi) + B * (xB - xA) * t2 / (A + B);

	double G00 = M_PI / std::sqrt(A * B);
	double G10 = C00 * G00;

	double Ix = std::exp(-Gx) * G10;
	return Ix;
}

double Ix_calc_psps(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double xA = (ai * xi + aj * xj) / (ai + aj);
	double xB = (ak * xk + al * xl) / (ak + al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double C00 = (xA - xi) + B * (xB - xA) * t2 / (A + B);
	double C00p = (xB - xk) + A * (xA - xB) * t2 / (A + B);
	double B00 = t2 / (2 * (A + B));

	double G00 = M_PI / std::sqrt(A * B);
	double G11 = (B00 + C00 * C00p) * G00;

	double Ix = std::exp(-Gx) * G11;
	return Ix;
}

double Ix_calc_ppss(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double xA = (ai * xi + aj * xj) / (ai + aj);
	double xB = (ak * xk + al * xl) / (ak + al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double C00 = (xA - xi) + B * (xB - xA) * t2 / (A + B);
	double B00 = t2 / (2 * (A + B));
	double B10 = 1 / (2 * A) - B * t2 / (2 * A * (A + B));

	double G00 = M_PI / std::sqrt(A * B);
	double G10 = C00 * G00;
	double G20 = (B10 + C00 * C00) * G00;

	double Ix = std::exp(-Gx) * (G20 + (xi - xj) * G10);
	return Ix;
}

double Ix_calc_ppps(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double xA = (ai * xi + aj * xj) / (ai + aj);
	double xB = (ak * xk + al * xl) / (ak + al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double C00 = (xA - xi) + B * (xB - xA) * t2 / (A + B);
	double C00p = (xB - xk) + A * (xA - xB) * t2 / (A + B);
	double B00 = t2 / (2 * (A + B));
	double B10 = 1 / (2 * A) - B * t2 / (2 * A * (A + B));

	double G00 = M_PI / std::sqrt(A * B);
	double G10 = C00 * G00;
	double G11 = (B00 + C00 * C00p) * G00;
	double G20 = (B10 + C00 * C00) * G00;
	double G21 = 2 * B00 * G10 + C00p * G20;

	double Ix = std::exp(-Gx) * (G21 + (xi - xj) * G11);
	return Ix;
}

double Ix_calc_pppp(double &t2, double &xi, double &xj, double &xk, double &xl, double &ai, double &aj, double &ak, double &al)
{
	double xA = (ai * xi + aj * xj) / (ai + aj);
	double xB = (ak * xk + al * xl) / (ak + al);
	double A = ai + aj;
	double B = ak + al;
	double Gx = ai * aj / (ai + aj) * (xi - xj) * (xi - xj) + ak * al / (ak + al) * (xk - xl) * (xk - xl);

	double C00 = (xA - xi) + B * (xB - xA) * t2 / (A + B);
	double C00p = (xB - xk) + A * (xA - xB) * t2 / (A + B);
	double B00 = t2 / (2 * (A + B));
	double B10 = 1 / (2 * A) - B * t2 / (2 * A * (A + B));
	double B01p = 1 / (2 * B) - A * t2 / (2 * B * (A + B));

	double G00 = M_PI / std::sqrt(A * B);
	double G10 = C00 * G00;
	double G01 = C00p * G00;
	double G11 = (B00 + C00 * C00p) * G00;
	double G20 = (B10 + C00 * C00) * G00;
	double G02 = (B01p + C00p * C00p) * G00;
	double G21 = 2 * B00 * G10 + C00p * G20;
	double G12 = 2 * B00 * G01 + C00 * G02;
	double G22 = B01p * G20 + 2 * B00 * G11 + C00p * G21;

	double Ix = std::exp(-Gx) * (G22 + (xi - xj) * G12 + (xk - xl) * G21 + (xi - xj) * (xk - xl) * G11);
	return Ix;
}
