project (wukong)

## We require the most recent version of cmake and automatically
## install the correct version when using the cmake lists
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

## Determine where wukong source code are defined
set(ROOT $ENV{WUKONG_ROOT})

## Use C++11 features
add_definitions(-std=c++11)

set(CMAKE_CXX_COMPILER /usr/local/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g")
set(BOOST_LIB "${ROOT}/deps/boost_1_58_0-install/lib" )
set(CUDA_LIB "/usr/local/cuda/lib64")
## Uncomment two lines below to enble HDFS support.
#add_definitions(-DHAS_HADOOP)
#target_link_libraries(wukong hdfs)

## Uncomment the line below to enable RDMA support.
add_definitions(-DHAS_RDMA)
#add_definitions(-DWUKONG_DEBUG)
#add_definitions(-DCUDA_DEBUG)

## Uncomment the line below to support versatile queries (e.g., ?S ?P ?O).
#add_definitions(-DVERSATILE)

## Set include path
include_directories(deps/hiredis)
include_directories(deps/boost_1_58_0-install/include)
include_directories(core)
include_directories(utils)

file(GLOB SOURCES "core/*.hpp" "utils/*.hpp")

# cuda_add_executable(wukong_gdr_test ${SOURCES} "test/test_gdr_adaptor.cpp")
# target_link_libraries(wukong_gdr_test zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a
    # ${BOOST_LIB}/libboost_serialization.a -L${CUDA_LIB} -lcudart)

# cuda_add_executable(test_rdma_send ${SOURCES} "test/test_rdma_send.cpp")
# target_link_libraries(test_rdma_send zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a
    # ${BOOST_LIB}/libboost_serialization.a -L${CUDA_LIB} -lcudart)

# cuda_add_executable(test_rdma_receiver ${SOURCES} "test/test_rdma_receiver.cpp")
# target_link_libraries(test_rdma_receiver zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a
    # ${BOOST_LIB}/libboost_serialization.a -L${CUDA_LIB} -lcudart)

# cuda_add_executable(test_gdr_send ${SOURCES} "test/test_gdr_send.cpp")
# target_link_libraries(test_gdr_send zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a
    # ${BOOST_LIB}/libboost_serialization.a -L${CUDA_LIB} -lcudart)

# cuda_add_executable(test_gdr_receiver ${SOURCES} "test/test_gdr_receiver.cpp")
# target_link_libraries(test_gdr_receiver zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a
    # ${BOOST_LIB}/libboost_serialization.a -L${CUDA_LIB} -lcudart)

## Wukong
cuda_add_executable(wukong ${SOURCES} "core/wukong.cpp" build/gpu_hash.o)
target_link_libraries(wukong zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a
    -L${CUDA_LIB} -lcudart)

