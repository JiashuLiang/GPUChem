#include "Jobinfo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include "Jobinfo.h"

// set default values of rem
std::map<std::string, std::string> rem_default { {"basis", "sto3g"}, {"scf_convergence", "7"},
           {"scf_max_iter", "100"}, {"scf_algorithm", "diis"}, {"method", "hf"}};

int JobInfo::read_from_file(std::string &fin, std::string &fout, std::string &scratch){
    fin_name = fin;
    fout_name = fout;
    scratch_dir = scratch;
    if (!std::filesystem::is_directory(scratch_dir)) {
        std::cerr << "Setup/Jobinfo.cpp: The directory specified by GPUChem_SCRATCH does not exist!\n ";
        return 1;
    }
    if(read_input(fin_name)){
        std::cerr << "Setup/Jobinfo.cpp: Failed to read from Input files\n ";
        return 1;
    }

    return 0;
}

int JobInfo::read_from_file(std::string &fin, std::string &fout){
    fin_name = fin;
    fout_name = fout;
    if(const char* env_p = std::getenv("GPUChem_SCRATCH")){
        scratch_dir = std::string(env_p);
        if (!std::filesystem::is_directory(scratch_dir)) {
            std::cerr << "Setup/Jobinfo.cpp: The directory specified by GPUChem_SCRATCH does not exist!\n ";
            return 1;
        }
    }
    else{
        std::cerr << "Setup/Jobinfo.cpp: Environment Variable GPUChem_SCRATCH does not exist!\n ";
        return 1;
    }

    if(read_input(fin_name)){
        std::cerr << "Setup/Jobinfo.cpp: Failed to read from Input files\n ";
        return 1;
    }
    return 0;
}

int JobInfo::read_from_file(std::string &fin){
    fin_name = fin;

    fout_name = fin_name.substr(0,fin_name.find_last_of('.'))+".out";
    if(const char* env_p = std::getenv("GPUChem_SCRATCH")){
        scratch_dir = std::string(env_p);
        if (!std::filesystem::is_directory(scratch_dir)) {
            std::cerr << "Setup/Jobinfo.cpp: The directory specified by GPUChem_SCRATCH does not exist!\n ";
            return 1;
        }
    }
    else{
        std::cerr << "Setup/Jobinfo.cpp: Environment Variable GPUChem_SCRATCH does not exist!\n ";
        return 1;
    }

    if(read_input(fin_name)){
        std::cerr << "Setup/Jobinfo.cpp: Failed to read from Input files\n ";
        return 1;
    }
    return 0;
}

int JobInfo::read_input(std::string &fname){
    fin_name = fname;

    // Check if the file exists
    if (!std::filesystem::exists(fin_name)) {
        std::cerr << "Error: Input File does not exist!\n";
        return 1;
    }

    // Read the input file line by line in lower case and store in a vector and find the lines containing $ as seperator
    std::ifstream input_file(fin_name);
    std::vector<std::string> lines;
    std::vector<int> symbol_idxs; // lines containing $ as seperator
    std::string line;
    while (std::getline(input_file, line)) {
        if (line.empty()) {
            continue; // skip empty line
        }
        // Convert all letter to lower case
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        lines.push_back(line);
        if (line.find("$") != std::string::npos) {
            symbol_idxs.push_back(lines.size() - 1);
        }
    }
    input_file.close();

    // Check if the input file is complete, containing at least $molecule, $end, $rem, and $end 
    if(symbol_idxs.size() < 4){
        std::cerr << "Error: Input File is not complete!\n";
        return 1;
    }

    // to find the start and end of $molecule and $rem
    auto mol_start= lines.end(), mol_end = lines.end(), rem_start = lines.end(), rem_end = lines.end();
    for(auto it= symbol_idxs.begin(); it != symbol_idxs.end(); ++it){
        if(lines[*it].find("$molecule") != std::string::npos ){
            mol_start = lines.begin() + *it + 1; // the next line of "$molecule"
            if(lines[*(it+1)].find("$end") != std::string::npos )
                mol_end = lines.begin() + *(it+1);
        }
        if (lines[*it].find("$rem") != std::string::npos){
            rem_start = lines.begin() + *it + 1;// the next line of "$rem"
            if(lines[*(it+1)].find("$end") != std::string::npos )
                rem_end = lines.begin() + *(it+1);
        }
    }

    // Read the molecule and section
    if (mol_start == lines.end() || mol_end == lines.end()) {
        std::cerr << "Error: Input File does not have a complete molecule section!\n";
        return 1;
    }else {
        m_molecule.convert_from_strvecs(mol_start, mol_end);
    }

    // Read the rem section
    if (rem_start == lines.end() || rem_end == lines.end()) {
        std::cerr << "Error: Input File does not have a complete rem section!\n";
        return 1;
    }else{
        for (auto it = rem_start; it != rem_end; ++it){
            std::istringstream iss(*it);
            std::string rem, value;
            if (!(iss >> rem >> value)){
                std::cerr << "Error: Input File does not have a correct rem section!\n";
                return 1;
            }
            // delete the comment part after !
            size_t exclamation = value.find('!');
            std::string nocomment_value = value.substr(0, exclamation);
            // remove all dashes
            nocomment_value.erase(std::remove(nocomment_value.begin(), nocomment_value.end(), '-'), nocomment_value.end()); // remove all dashes
            REM_map.insert_or_assign(rem, nocomment_value); // replace the value if the key already exists
        }
    }


    return 0;
}


std::string JobInfo::GetRem(const std::string &key){
    // find whether the key exists, return the value if it exists or it has default value, otherwise throw an error
    auto it = REM_map.find(key);
    if (it != REM_map.end()){
        return it->second;
    }else{
        auto it_default = rem_default.find(key);
        if (it_default != rem_default.end()){
            return it_default->second;
        }else
            std::cerr << "Error: The key " << key <<" does not exist in the REM section!\n";
            //return empty string
            return "";
    }
}


void JobInfo::PrintRemInfo(){
    for(const auto& elem : REM_map)
        std::cout << elem.first << " " << elem.second << "\n";
}
