#ifndef JOBINFO_H
#define JOBINFO_H

#include "molecule.h"
#include <map>
#include <string>

class JobInfo{
    public:
        std::string fin_name;
        std::string fout_name;
        std::string scratch_dir;

        Molecule m_molecule;
        std::map<std::string, std::string> REM_map;  


        JobInfo() = default;
        
        int read_from_file(std::string &fin); //Read from file, output to fout, using scratch directory specified by environment variable "GPUChem_SCRATCH"
        int read_from_file(std::string &fin, std::string &fout); //Read from file and output to fout, using scratch directory specified by environment variable "GPUChem_SCRATCH"
        int read_from_file(std::string &fin, std::string &fout, std::string &scratch); //Read from file and output to fout, using specified scratch directory
        int read_input(std::string &fname);

        void PrintRemInfo();
        std::string GetRem(const std::string &key);
};

#endif // JOBINFO_H
