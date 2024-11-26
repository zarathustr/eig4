cmake_minimum_required(VERSION 3.24)
project(eig4)

set(CMAKE_CXX_STANDARD 11)

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIRS} )

add_executable(eig4 main.cpp eig4.cpp)
target_compile_options(eig4 PRIVATE -Os)
