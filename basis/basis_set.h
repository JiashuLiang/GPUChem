#ifndef BASIS_SET_H
#define BASIS_SET_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <armadillo>

class BasisShell{
    public:
    std::string Shell_type;
    int num_primitives;
    double scale_factor;
    arma::vec exponents;
    arma::mat coefficients; // the outer is the number of contractions, the inner is the number of different angular momentums

    BasisShell() = default;
    BasisShell(std::string shell_type, int num_p, arma::vec expo, arma::mat coef):
        Shell_type(shell_type), num_primitives(num_p), exponents(expo), coefficients(coef){}
    void read_from_strvec(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end);
    BasisShell(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end);
    void PrintShell() const;
};


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