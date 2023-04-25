# GPUChem
A Quantum Chemistry package to run on GPU

To compile on NERSC, first install armadillo and then download the package
```
git clone git@github.com:JiashuLiang/GPUChem.git
```
Then configure it
```
cd GPUChem
mkdir build
cd build
ARMADILLO_DIR=${ARMADILLO_INSTALL_PATH} cmake -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_CXX_COMPILER=CC ..
```

If you are in mp54 project, you can use the armadillo downloaded by me 
```
ARMADILLO_DIR=/global/cfs/cdirs/mp54/armadillo-12.2.0/build  cmake -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_CXX_COMPILER=CC ..
```
Then make the tests
```
make GPUChem_tests
```

Before running the tests, make sure defining environment variable GPUChem_SCRATCH and GPUChem_aux (aux has been downloaded in the pacakage). For example,
```
export GPUChem_SCRATCH=$PSCRATCH
export GPUChem_aux={GPUChem_PATH}/aux
```

There are three tests for now. You can run each by
```
ctest -R Setup_test -V
ctest -R basis_test -V
ctest -R SCF_test -V
```
