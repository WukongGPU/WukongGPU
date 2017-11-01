#!/bin/sh
mkdir -p ../build;
cd ../build;
nvcc -c ../core/gpu_hash.cu -o gpu_hash.o -arch=sm_35 -O3 
cmake ..;
make;
cd ../scripts;
