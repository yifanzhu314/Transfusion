cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(ElasticFusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")  



set(Torch_DIR  /home/zyf-lab/program/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS}) 
message(STATUS "Torch library status:") 
message(STATUS "    version: ${TORCH_VERSION}") 
message(STATUS "    libraries: ${TORCH_LIBS}") 
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(efusion REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(RealSense2 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "librealsense found: ${REALSENSE_FOUND}")
option(WITH_REALSENSE "Build with Intel RealSense2 support?" ${REALSENSE2_FOUND})

include_directories(${REALSENSE2_INCLUDE_DIR})
add_definitions(-DWITH_REALSENSE)
set(EXTRA_LIBS ${EXTRA_LIBS} ${REALSENSE2_LIBRARY})

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EFUSION_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})


file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)
  




SET(CMAKE_BUILD_TYPE "Debug")     
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")      
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  

add_executable(ElasticFusion 
               ${srcs}
               ${tools_srcs}
               ${hdrs}
               ${tools_hdrs}
)

target_link_libraries(ElasticFusion                  
                      "${ZLIB_LIBRARY}"
                      "${Pangolin_LIBRARIES}"
                      "${CUDA_LIBRARIES}"
                      "${EXTRA_LIBS}"
                      "${EFUSION_LIBRARY}"
                      "${OPENNI2_LIBRARY}"
                      "${SUITESPARSE_LIBRARIES}"
                      "${BLAS_LIBRARIES}"
                      "${LAPACK_LIBRARIES}"
                      "${TORCH_LIBRARIES}"                                                  
                      "${OpenCV_LIBS}"   
                                          
)


INSTALL(TARGETS ElasticFusion
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
set_property(TARGET ElasticFusion  PROPERTY CXX_STANDARD 14)
