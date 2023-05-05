set(TESTS
    Setup_test
    basis_test
    Hamiltonian_test
    SCF_test
)
include_directories(${ARMADILLO_INCLUDE_DIRS})

foreach(TEST ${TESTS})
    set(TGT GPUChem_${TEST})
    list(APPEND TEST_TGTS ${TGT})
    add_executable(${TGT} ${TEST}.cpp)
    set_target_properties(${TGT} PROPERTIES CUDA_ARCHITECTURES "80")
    target_link_libraries(${TGT} setup basis integral scf ${ARMADILLO_LIBRARIES})
    add_test(NAME "${TEST}" COMMAND ${TGT}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach(TEST)
add_custom_target(GPUChem_tests DEPENDS ${TEST_TGTS})