cmake_minimum_required(VERSION 3.5)
project(LDPC)

find_package(OpenCL REQUIRED)
#find_package(Boost 1.54 REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
include_directories(SYSTEM ${Boost_HEADERS})
#add_library(lib src/matrix_generate.cpp)
#target_include_directories(lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# add_executable(LDPC_bit_flipping src/LDPC_bit_flipping.cpp)
# target_link_libraries(LDPC_bit_flipping ${Boost_HEADERS} OpenCL::OpenCL)

add_executable(LDPC_min_sum_naive src/LDPC_min_sum_naive.cpp)
target_link_libraries(LDPC_min_sum_naive ${Boost_HEADERS} OpenCL::OpenCL)

add_executable(LDPC_min_sum_byindex src/LDPC_min_sum_byindex.cpp)
target_link_libraries(LDPC_min_sum_byindex ${Boost_HEADERS} OpenCL::OpenCL)

add_executable(minsum_naive src/minsum_opencl_naive.cpp)
target_link_libraries(minsum_naive OpenCL::OpenCL)
set(kernels KernelHS_naive.cl
            KernelHS_byindex.cl
            KernelVS_naive.cl
            KernelVS_naive.cl
            KernelCheck_naive.cl
            KernelCheck_byindex.cl
            KernelLLR.cl)
foreach(kernel IN ITEMS ${kernels})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cl/${kernel} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
# add_executable(boo src/boo.cpp)
# target_link_libraries(boo ${Boost_HEADERS} OpenCL::OpenCL)

#-DBoost_HEADERS=/home/rinat/boost_1_75_0
