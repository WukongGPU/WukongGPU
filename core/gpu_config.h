#pragma once
#include <cuda_runtime.h>


#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : code: %d, %s.\n",        \
                __FILE__, __LINE__, err, cudaGetErrorString(err) );              \
        assert(false);                                                       \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);


#define GPU_ASSERT(ans) { gpuCheckResult((ans), __FILE__, __LINE__); }
inline void gpuCheckResult(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: code:%d, %s %s:%d\n", code, cudaGetErrorString(code), file, line);
      if (abort) assert(false);
   }
}


#define pf(x) cout <<#x<<": "<<x<<endl;

const int blocksize = 16; 

const int ASSOCIATIVITY = 8;


// for RCache
#define GPU_MAX_ELEM 20000000
#define GPU_BUF_SIZE(ele_sz) (ele_sz * GPU_MAX_ELEM)
#define MAX_TABLE_ROW(ncol) (GPU_DRAFT_BUF_SIZE / ncol)
