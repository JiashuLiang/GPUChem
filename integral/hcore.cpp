#define _USE_MATH_DEFINES
#include "hcore.h"
#include <basis/molecule_basis.h>
#include <basis/util.h>
#include <armadillo>
#include <math.h>
#include <assert.h>

#define max(a,b) ((a)>(b)?(a):(b))

// vnn.C
int eval_OVmat(Molecule_basis& system, arma::mat &S_mat){
    const int nbsf = system.mAOs.size();
    S_mat.set_size(nbsf,nbsf);
    S_mat.zeros();

    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices;
    
    // Sort AOs by type
    size_t p_start_ind = sort_AOs(system.mAOs, sorted_AOs, sorted_indices);

    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    // Perform construction of S, sorted into blocks of ss, sp, ps,pp
    construct_S(S_mat, sorted_AOs, p_start_ind);

    // return H_mat to its original order.
    S_mat = S_mat(undo_sorted_indices, undo_sorted_indices);

    return 0;
}


int eval_Hcoremat(Molecule_basis& system, arma::mat &H_mat){
    const int nbsf = system.mAOs.size();
    H_mat.set_size(nbsf,nbsf);
    arma::mat T_mat(nbsf,nbsf), V_mat(nbsf,nbsf);
    
    T_mat.zeros();
    V_mat.zeros();

    std::vector<AO> sorted_AOs;
    arma::uvec sorted_indices;
    
    // Sort AOs by type
    size_t p_start_ind = sort_AOs(system.mAOs, sorted_AOs, sorted_indices);

    arma::uvec undo_sorted_indices = arma::sort_index(sorted_indices);

    // Perform construction of H, sorted into blocks of ss, sp, ps,pp
    construct_V(V_mat, sorted_AOs, p_start_ind, system.m_mol);
    construct_T(T_mat, sorted_AOs, p_start_ind);


    
    
    H_mat = T_mat + V_mat;

    std::cout << "Printing T mat "<<std::endl;
    T_mat.print();
    std::cout << "Printing V mat "<<std::endl;
    V_mat.print();
    // return H_mat to its original order.
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

void construct_S(arma::mat &Smat, std::vector<AO> &mAOs, size_t p_start_ind){
    // Handle ss, then sp, then pp.
    // Might be inefficient for small s_orbs.size() and p_orbs.size()
    
    // ss
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Smat(mu,nu) = eval_Smunu(mAOs[mu], mAOs[nu]);
        }
    }
    // ps
    for (size_t mu = p_start_ind; mu < Smat.n_rows; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Smat(mu,nu) = eval_Smunu(mAOs[mu], mAOs[nu]);
        }
    }
    // sp
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = p_start_ind; nu < Smat.n_cols; nu++){
            Smat(mu,nu) = eval_Smunu(mAOs[mu], mAOs[nu]);
        }
    }
    // pp
    for (size_t mu = p_start_ind; mu < Smat.n_rows; mu++){
        for (size_t nu = p_start_ind; nu < Smat.n_cols; nu++){
            Smat(mu,nu) = eval_Smunu(mAOs[mu], mAOs[nu]);
        }
    }

}

void construct_V(arma::mat &Vmat, std::vector<AO> &mAOs, size_t p_start_ind, const Molecule &mol){
    // Handle ss, then sp, then pp.
    // Might be inefficient for small s_orbs.size() and p_orbs.size()
    
    // ss
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Vmat(mu,nu) = eval_Vmunu(mAOs[mu], mAOs[nu], mol);
        }
    }
    // ps
    for (size_t mu = p_start_ind; mu < Vmat.n_rows; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Vmat(mu,nu) = eval_Vmunu(mAOs[mu], mAOs[nu], mol);
        }
    }
    // sp
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = p_start_ind; nu < Vmat.n_cols; nu++){
            Vmat(mu,nu) = eval_Vmunu(mAOs[mu], mAOs[nu], mol);
        }
    }
    // pp
    for (size_t mu = p_start_ind; mu < Vmat.n_rows; mu++){
        for (size_t nu = p_start_ind; nu < Vmat.n_cols; nu++){
            Vmat(mu,nu) = eval_Vmunu(mAOs[mu], mAOs[nu], mol);
        }
    }

}
void construct_T(arma::mat &Tmat, std::vector<AO> &mAOs, size_t p_start_ind){
    // for (size_t mu = 0; mu < Tmat.n_rows; mu++){
    //     for (size_t nu = 0; nu < Tmat.n_cols; nu++){
    //         Tmat(mu,nu) = eval_Tmunu(mAOs[mu], mAOs[nu]);
    //     }
    // }

    // Handle ss, then sp, then pp.
    // Might be inefficient for small s_orbs.size() and p_orbs.size()
    
    // ss
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Tmat(mu,nu) = eval_Tmunu(mAOs[mu], mAOs[nu]);
        }
    }
    // ps
    for (size_t mu = p_start_ind; mu < Tmat.n_rows; mu++){
        for (size_t nu = 0; nu < p_start_ind; nu++){
            Tmat(mu,nu) = eval_Tmunu(mAOs[mu], mAOs[nu]);
        }
    }
    // sp
    for (size_t mu = 0; mu < p_start_ind; mu++){
        for (size_t nu = p_start_ind; nu < Tmat.n_cols; nu++){
            Tmat(mu,nu) = eval_Tmunu(mAOs[mu], mAOs[nu]);
        }
    }
    // pp
    for (size_t mu = p_start_ind; mu < Tmat.n_rows; mu++){
        for (size_t nu = p_start_ind; nu < Tmat.n_cols; nu++){
            Tmat(mu,nu) = eval_Tmunu(mAOs[mu], mAOs[nu]);
        }
    }

}



double eval_Smunu(AO &mu, AO &nu){
    assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?
    
    size_t mu_no_primitives = mu.alpha.size();
    size_t nu_no_primitives = nu.alpha.size();
    
    double total = 0.0;
    int l1 = mu.lmn(0);
    int m1 = mu.lmn(1);
    int n1 = mu.lmn(2);
    
    int l2 = nu.lmn(0);
    int m2 = nu.lmn(1);
    int n2 = nu.lmn(2);
    arma::vec & A = mu.R0;
    arma::vec & B = nu.R0;

    for (size_t mup = 0; mup < mu_no_primitives; mup++){
        double alpha = mu.alpha(mup);
        double d_kmu = mu.d_coe(mup);

        for (size_t nup = 0; nup < nu_no_primitives; nup++){
            double beta = nu.alpha(nup);
            double d_knu = nu.d_coe(nup);
            total +=  d_knu * d_kmu * overlap(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta);
        }
    }
    return total;

}


double eval_Tmunu(AO &mu, AO &nu){
    assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?
    
    size_t mu_no_primitives = mu.alpha.size();
    size_t nu_no_primitives = nu.alpha.size();
    
    double total = 0.0;
    int l1 = mu.lmn(0);
    int m1 = mu.lmn(1);
    int n1 = mu.lmn(2);
    
    int l2 = nu.lmn(0);
    int m2 = nu.lmn(1);
    int n2 = nu.lmn(2);
    arma::vec & A = mu.R0;
    arma::vec & B = nu.R0;

    for (size_t mup = 0; mup < mu_no_primitives; mup++){
        double alpha = mu.alpha(mup);
        double d_kmu = mu.d_coe(mup);

        for (size_t nup = 0; nup < nu_no_primitives; nup++){
            double beta = nu.alpha(nup);
            double d_knu = nu.d_coe(nup);
            total +=  d_knu * d_kmu * kinetic(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta);
        }
    }
    return total;
}

double eval_Vmunu(AO &mu, AO &nu, const Molecule &mol){
    // nuclear attraction
    assert(mu.alpha.size()==mu.d_coe.size() && nu.alpha.size()==nu.d_coe.size()); // This should be true?
    
    size_t mu_no_primitives = mu.alpha.size();
    size_t nu_no_primitives = nu.alpha.size();
    
    double total = 0.0;
    int l1 = mu.lmn(0);
    int m1 = mu.lmn(1);
    int n1 = mu.lmn(2);
    
    int l2 = nu.lmn(0);
    int m2 = nu.lmn(1);
    int n2 = nu.lmn(2);
    arma::vec A = mu.R0;
    arma::vec B = nu.R0;

    for (size_t c = 0; c < mol.mAtoms.size(); c++){
        arma::vec C = mol.mAtoms[c].m_coord; // coordinates of the atom
        for (size_t mup = 0; mup < mu_no_primitives; mup++){
            double alpha = mu.alpha(mup);
            double d_kmu = mu.d_coe(mup);

            for (size_t nup = 0; nup < nu_no_primitives; nup++){
                double beta = nu.alpha(nup);
                double d_knu = nu.d_coe(nup);
                total +=  d_knu * d_kmu * nuclear_attraction(A,  l1,  m1, n1, alpha, B, l2, m2, n2, beta, C);
            }
        }
    }
    return total;
}


double gammln(double xx){
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

void gser(double *gamser, double a, double x, double *gln) {
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
        if (x < 0.0) throw std::runtime_error("x less than 0 in routine gser");
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
        throw std::runtime_error("a too large, ITMAX too small in routine gser");
        return;
    }
}

void gcf(double *gammcf, double a, double x, double *gln){
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
    if (i > ITMAX) throw std::runtime_error("a too large, ITMAX too small in gcf");
    *gammcf = exp(-x+a*log(x)-(*gln))*h; //Put factors in front.
}

double gammp(double a, double x){
    // Returns the incomplete gamma function P (a, x). From Numerical Recipes, section 6.1 and 6.2
    double gam, gamc, gln;
    if (x < 0.0 || a <= 0.0) throw std::runtime_error("Invalid arguments in routine gammp");

    if (x < (a+1.0)) {// Use the series representation.
        gser(&gam,a,x,&gln);
    } else { //Use the continued fraction representation
        gcf(&gamc,a,x,&gln);
        gam = 1-gamc;
        
    }
    return exp(gln)*gam;
}

double Fgamma(int m, double x){
    // Incomplete Gamma Function
    double SMALL=1e-12;
    double m_d = (double) m; // convert to double explicitly, prolly notneeded
    x = max(x,SMALL);
    // std::cout<<"-m_d-0.5 --" <<(-m_d-0.5)<<std::endl;
    // std::cout<<"-m-0.5 --" <<(-m-0.5)<<std::endl;
    return 0.5*pow(x,-m_d-0.5)*gammp(m_d+0.5,x);
}

int factorial (int n){
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

int nCk (int n, int k){
    return factorial(n)/(factorial(k) * factorial(n-k));
}

arma::vec gaussian_product_center(double alpha, arma::vec &A, double beta, arma::vec &B){
    //computes the new gaussian center P.
    return (alpha*A + beta*B)/(alpha + beta);
}

double poly_binom_expans_terms(int n, int ia, int ib, double PminusA_1d, double PminusB_1d){
    // computes the binomial expansion for the terms where ia + ib = n.
    double total = 0.0;

    for (int t = 0; t < n + 1; t++){
        if (n - ia <= t && t <= ib){
            total += Combination(ia, n-t) * Combination (ib,t) * pow(PminusA_1d, ia-n+t) * pow(PminusB_1d, ib-t);
        }
    }
    return total;
}

double overlap_1d(int l1, int l2, double PminusA_1d, double PminusB_1d, double gamma){
    double total  = 0.0;
    for (int i = 0; i < (1+ int(std::floor((l1+l2)/2))); i++){
        total += poly_binom_expans_terms(2*i, l1, l2,PminusA_1d, PminusB_1d )* DoubleFactorial(2*i-1)/pow(2*gamma, i);
    }
    return total;

}

double overlap(arma::vec A,  int l1, int m1, int n1,double alpha, arma::vec B, int l2, int m2, int n2,double beta ){
    double gamma = alpha + beta;
    arma::vec P = gaussian_product_center(alpha, A, beta, B);

    double prefactor = pow(M_PI/gamma,1.5) * exp(-alpha * beta * pow(arma::norm(A-B),2)/gamma);

    double sx = overlap_1d(l1,l2,P(0)-A(0),P(0)-B(0),gamma);
    double sy = overlap_1d(m1,m2,P(1)-A(1),P(1)-B(1),gamma);
    double sz = overlap_1d(n1,n2,P(2)-A(2),P(2)-B(2),gamma);
    return prefactor * sx * sy * sz;
}

double kinetic(arma::vec A,int l1, int m1, int n1,double alpha, arma::vec B, int l2, int m2, int n2, double beta){
    // Formulation from JPS (21) 11, Nov 1966 by H Taketa et. al

    double term0 = beta*(2*(l2+m2+n2)+3)*overlap(A,l1,m1,n1,alpha,B,l2,m2,n2,beta);

    double term1 = -2*pow(beta,2)*(overlap(A,l1,m1,n1,alpha, B,l2+2,m2,n2,beta) +\
                            overlap(A,l1,m1,n1,alpha, B,l2,m2+2,n2,beta) +\
                            overlap(A,l1,m1,n1,alpha, B,l2,m2,n2+2,beta));
    double term2 = -0.5*(l2*(l2-1)*overlap(A,l1,m1,n1,alpha, B,l2-2,m2,n2,beta) +\
                  m2*(m2-1)*overlap(A,l1,m1,n1,alpha, B,l2,m2-2,n2,beta) +\
                  n2*(n2-1)*overlap(A,l1,m1,n1,alpha, B,l2,m2,n2-2,beta));

    return term0 + term1 + term2;
}



double A_term(int i, int r, int u, int l1, int l2,double PAx, double PBx, double CPx, double gamma){

    return pow(-1,i)*poly_binom_expans_terms(i,l1,l2,PAx,PBx)*\
           pow(-1,u)*factorial(i)*pow(CPx,i-2*r-2*u)*\
           pow(0.25/gamma,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u);
}

arma::vec A_tensor(int l1, int l2, double PA, double PB, double CP, double g){
    int Imax = l1+l2+1;
    arma::vec A(Imax);
    A.zeros();
    for (int i = 0; i < Imax; i++){
        for (int r = 0; r < int(std::floor(i/2)+1); r++){
            for (int u = 0; u < int(floor((i-2*r)/2)+1); u++){
                int  I = i - 2*r - u;
                A[I] = A[I] + A_term(i,r,u,l1,l2,PA,PB,CP,g);
            }
        }
    }

    return A;

}
double nuclear_attraction(arma::vec &A,int l1, int m1, int n1,double alpha, arma::vec &B, int l2, int m2, int n2,double beta, arma::vec &C){
    // Formulation from JPS (21) 11, Nov 1966 by H Taketa et. al
    
    double gamma = alpha + beta;

    arma::vec P = gaussian_product_center(alpha,A,beta,B);
    double rab2 = pow(arma::norm(A-B),2);
    double rcp2 = pow(arma::norm(C-P),2);

    arma::vec dPA = P-A;
    arma::vec dPB = P-B;
    arma::vec dPC = P-C;

    arma::vec Ax = A_tensor(l1,l2,dPA(0),dPB(0),dPC(0),gamma);
    arma::vec Ay = A_tensor(m1,m2,dPA(1),dPB(1),dPC(1),gamma);
    arma::vec Az = A_tensor(n1,n2,dPA(2),dPB(2),dPC(2),gamma);

    double total = 0.0;
    for (int I = 0; I < l1+l2+1; I++)
        for (int J = 0; J < m1+m2+1; J++)
            for (int K = 0; K < n1+n2+1; K++)
                total += Ax(I)*Ay(J)*Az(K)*Fgamma(I+J+K,rcp2*gamma);
                
    double val= -2*M_PI/gamma*exp(-alpha*beta*rab2/gamma)*total;
    return val;
}