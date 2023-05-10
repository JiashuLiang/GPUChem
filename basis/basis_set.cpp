#include "basis_set.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <filesystem>

void BasisShell::read_from_strvec(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end){
    std::istringstream iss(*shell_start);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
    //Read the first line
    // Format: Shell_type num_primitives scale_factor
    Shell_type = tokens[0];
    num_primitives = std::stoi(tokens[1]);
    scale_factor = std::stod(tokens[2]);

    std::vector<double> input_exponents;
    std::vector<std::vector<double>> input_coefficients;
    //Read the rest lines
    // Format: exponent d_coef1_firstshell d_coef1_secondshell ...
    //                  d_coef2_firstshell d_coef2_secondshell ...
    //                  d_coef3_firstshell d_coef3_secondshell ...
    //                  ... (num_primitives lines)
    for (size_t i = 0; i < num_primitives; i++){
        std::istringstream issi(*(shell_start + i + 1));
        std::vector<std::string> tokens{std::istream_iterator<std::string>{issi}, std::istream_iterator<std::string>{}};
        input_exponents.push_back(std::stod(tokens[0]));
        std::vector<double> coef;
        for (size_t j = 1; j < tokens.size(); j++){
            coef.push_back(std::stod(tokens[j]));
        }
        input_coefficients.push_back(coef);
    }
    // Check whether the number of contraction is correct
    if (input_coefficients.size() != num_primitives || input_exponents.size() != num_primitives){
        std::cerr << "Error: the number of contraction is not correct!\n";
        return;
    }
    // convert exponents from the std::vector to arma::vec
    this->exponents = arma::vec(input_exponents);
    // convert coefficients from the std::vector<std::vector<double>> to arma::mat
    this->coefficients = arma::mat(input_coefficients.size(), input_coefficients[0].size());
    for (size_t i = 0; i < input_coefficients.size(); i++){
        for (size_t j = 0; j < input_coefficients[0].size(); j++){
            this->coefficients(i, j) = input_coefficients[i][j];
        }
    }

    
}

BasisShell::BasisShell(std::vector<std::string>::iterator shell_start, std::vector<std::string>::iterator shell_end){
    read_from_strvec(shell_start, shell_end);
}

void BasisShell::PrintShell() const{
    std::cout << Shell_type << " " << num_primitives << " " << scale_factor << std::endl;
    for (size_t i = 0; i < num_primitives; i++){
        std::cout << exponents(i) << " ";
        for (size_t j = 0; j < coefficients.n_cols; j++){
            std::cout << coefficients(i,j) << " ";
        }
        std::cout << std::endl;
    }
}



BasisSet::BasisSet(const std::string& BasisName) {
    std::string aux;
    // Check whether the environment variable GPUChem_aux is set
    if(const char* env_p = std::getenv("GPUChem_aux")){
        aux = std::string(env_p);
        if (!std::filesystem::is_directory(aux)) {
            throw std::runtime_error("basis/basis_set.cpp: The directory specified by GPUChem_aux does not exist!");
        }
    }

    std::string filename = aux + "/basis/" + BasisName + ".bas"; // The path of the basis set file
    // std::cout << filename << std::endl;
    readBasisFile(filename);
    // Compare whether BasisName is same as basisName_
    if (BasisName != basisName_){
        throw std::runtime_error("basis/basis_set.cpp: Basis name is not correct! Require "+ BasisName + ", but get " + basisName_ );
    }
}

void BasisSet::readBasisFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open basis set file: " << filename << std::endl;
        return;
    }

    std::string line;

    // Read the basis name from the first line
    std::getline(file, line);
    if (line.find("BASIS=") != std::string::npos) {
        basisName_ = line.substr(line.find("=") + 1);
        // Convert the basis name to lowercase and remove the dash in the name
        std::transform(basisName_.begin(), basisName_.end(), basisName_.begin(), ::tolower);
        basisName_.erase(std::remove(basisName_.begin(), basisName_.end(), '-'), basisName_.end());
        basisName_.erase(std::remove(basisName_.begin(), basisName_.end(), '"'), basisName_.end());
    } else{
        std::cerr << "Error: Basis set file does not have a correct basis name!\n";
        std::cerr << line;
        return;
    }

    // Read the rest of the file in lower case and record the line number of the element name
    std::vector<size_t> symbol_idxs; // lines when the basis set starts
    symbol_idxs.push_back(0);
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // skip empty line
        }
        // Convert all letter to lower case
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        lines.push_back(line);
        if (line.find("****") != std::string::npos) {
            symbol_idxs.push_back(lines.size());
        }
    }
    file.close();

    // begin get the basis info one element by one element
    std::string currentElement;
    for(size_t i = 0; i < symbol_idxs.size() - 1; i++) {
        std::istringstream iss(lines[symbol_idxs[i]]);
        //read the element name as the first non-spacing string
        if (!(iss >> currentElement)){
            std::cerr << "basis/basis_set.cpp: There is some problem with element name.\n ";
            std::cerr << lines[symbol_idxs[i]];
        }

        std::vector<BasisShell> currentBasis;
        // begin reading the shell info for the current element
        for(size_t j = symbol_idxs[i] + 1; j < symbol_idxs[i + 1]; j++) {
            //check if the first non-spacing string is a letter (shell type) or number
            std::istringstream iss(lines[j]);
            std::string first_str;
            if (!(iss >> first_str)){
                std::cerr << "basis/basis_set.cpp: There is some problem with shell info.\n ";
                std::cerr << lines[j];
            }
            if (std::isalpha(first_str[0])){
                // this line is the start of a new shell
                std::vector<std::string>::iterator shell_start = lines.begin() + j;
                std::vector<std::string>::iterator shell_end = lines.begin() + j + 1;
                while (shell_end != lines.end()){
                    std::istringstream iss(*(shell_end));
                    std::string first_str;
                    if (!(iss >> first_str)){
                        std::cerr << "basis/basis_set.cpp: There is some problem with shell info.\n ";
                        std::cerr << lines[j];
                    }
                    if (std::isalpha(first_str[0])){
                        break;
                    }
                    shell_end++;
                }
                // Use the iterator to construct the BasisShell
                BasisShell currentShell(shell_start, shell_end);
                currentBasis.push_back(currentShell);
            }

        }
        //store the basis info for the current element
        basisSets_[currentElement] = currentBasis;
    }

    // Check whether the basis set is read correctly
    if (basisSets_.size() == 0){
        std::cerr << "Error: the basis set is not read correctly!\n";
        return;
    }

    // Read the supported elements from the line next to "Elements supported" and before "---------------------"
    size_t sup_start = lines.size(), sup_end = lines.size();
    for (size_t i = symbol_idxs.back(); i < lines.size(); i++){
        if (lines[i].find("elements supported") != std::string::npos)
            sup_start = i;
        if (lines[i].find("---------------------") != std::string::npos)
            sup_end = i;
    }
    //Begin reading the supported elements
    for (size_t i = sup_start + 1; i < sup_end; i++){
        std::istringstream iss(lines[i]);
        // read all non-spacing strings in the line
        std::string first_str;
        while (iss >> first_str){
            SupportedElements_.push_back(first_str);
        }
    }

    // Check whether the supported elements are all in the keys of basisSets_
    for (auto it : SupportedElements_){
        if (basisSets_.find(it) == basisSets_.end()){
            std::cerr << "Error: the supported elements are not all in the keys of basisSets_!\n";
            return;
        }
    }

}

void BasisSet::setElementBasis(const std::string& element, const std::vector<BasisShell>& basis) {
    basisSets_[element] = basis;
}

std::vector<BasisShell> BasisSet::getElementBasis(const std::string& element) const{
    if (!isElementSupported(element)) {
        std::cerr << "Element " << element << " is not supported in this basis set" << std::endl;
    }
    return basisSets_.at(element);
}

bool BasisSet::isElementSupported(const std::string& element) const {
    return basisSets_.count(element) != 0;
}

std::vector<std::string> BasisSet::getSupportedElements() const{
    return SupportedElements_;
}

void BasisSet::PrintBasis() const{
    for (auto it : basisSets_){
        std::cout << it.first << std::endl;
        for (auto shell: it.second)
            shell.PrintShell();
        std::cout << std::endl;
    }
}