#ifndef BASIS_SET_H
#define BASIS_SET_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <armadillo>

// This class is used to store the information of a basis shell
// Components of a BasisSet class
class BasisShell{
    public:
    std::string Shell_type; // S, P, SP, D, F, 
    int num_primitives; // number of primitive gaussians
    double scale_factor; // scale factor for the whole shell
    arma::vec exponents; // the exponents of the primitive gaussians
    arma::mat coefficients; // one column is the contraction coefficients for one kind of shell

    BasisShell() = default;
    BasisShell(std::string shell_type, int num_p, arma::vec expo, arma::mat coef):
        Shell_type(shell_type), num_primitives(num_p), exponents(expo), coefficients(coef){}
    
    // using read_from_strvec to construct a BasisShell
    BasisShell(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end);
    
    // read the basis shell from a string vectors, usually read from a block of a basis set file
    void read_from_strvec(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end);
    void PrintShell() const;
};


// This class is used to store the basis set information
// Usually read from a basis set file in the Q-Chem format
class BasisSet {
public:
    BasisSet() = default;
    BasisSet(const std::string& BasisName);
    void readBasisFile(const std::string& filename);
    void setElementBasis(const std::string& element, const std::vector<BasisShell>& basis);
    std::vector<BasisShell> getElementBasis(const std::string& element) const;
    bool isElementSupported(const std::string& element) const;
    std::vector<std::string> getSupportedElements() const;
    void PrintBasis() const;
private:
    std::string basisName_;
    std::vector<std::string> SupportedElements_;
    std::map<std::string, std::vector<BasisShell>> basisSets_;

};



#endif // BASIS_SET_H