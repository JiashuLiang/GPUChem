#include <stdlib.h>
#include <stdexcept>
#include <stdio.h>
#include <armadillo>
#include <vector>
// #include "AO.h"
// #include "CNDO.h"
#include <Setup/Jobinfo.h>

using namespace std;



int TestFiles(std::string fin, std::string fout, std::string  scratch){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout, scratch))
      return 1;

    MyJob.m_molecule.PrintMoleculeInfo();
    MyJob.PrintRemInfo();
    return 0;
}

int TestFiles(std::string fin, std::string fout){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin, fout))
      return 1;

    MyJob.m_molecule.PrintMoleculeInfo();
    MyJob.PrintRemInfo();
    return 0;
}

int TestFiles(std::string fin){
    JobInfo MyJob;
    std::cout << "Input: " << fin << "\n";

    if(MyJob.read_from_file(fin))
      return 1;

    MyJob.m_molecule.PrintMoleculeInfo();
    MyJob.PrintRemInfo();
    return 0;
}




int main(int argc, char *argv[])
{
  return
    TestFiles("C2H4/C2H4.in", "C2H4/C2H4.out")|
    TestFiles("C2H4/C2H4.in")|
    0;
}
