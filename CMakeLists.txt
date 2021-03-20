cmake_minimum_required(VERSION 3.5)
project(LDPC)

find_package(OpenCL REQUIRED)
#find_package(Boost 1.54 REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS -> "${OpenCL_LIBRARIES}")
add_executable(LDPC_bit_flipping src/LDPC_bit_flipping.cpp)
target_link_libraries(LDPC_bit_flipping ${Boost_HEADERS} OpenCL::OpenCL)

add_executable(LDPC_min_sum src/LDPC_min_sum.cpp)
target_link_libraries(LDPC_min_sum ${Boost_HEADERS} OpenCL::OpenCL)

add_executable(boo src/boo.cpp)
target_link_libraries(boo ${Boost_HEADERS} OpenCL::OpenCL)


