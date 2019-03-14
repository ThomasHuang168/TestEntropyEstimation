cmake_minimum_required(VERSION 2.8)

project( Test_openCV )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONs OFF)

include_directories(.)
find_package( OpenCV REQUIRED )
add_executable( Test_openCV test.cpp )
target_link_libraries( Test_openCV ${OpenCV_LIBS} )
