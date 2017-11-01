#pragma once


#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : code: %d, %s.\n",        \
                __FILE__, __LINE__, err, cudaGetErrorString(err) );              \
        assert(false);                                                       \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#define pf(x) cout <<#x<<": "<<x<<endl;

const int blocksize = 16; 

const int ASSOCIATIVITY = 8;


// for RCache
#define GPU_MAX_ELEM 40000000
#define NGPU_SHARDS 100
#define GPU_BUF_SIZE(ele_sz) (ele_sz * GPU_MAX_ELEM)
#define MAX_TABLE_ROW(ncol) (GPU_DRAFT_BUF_SIZE / ncol)
