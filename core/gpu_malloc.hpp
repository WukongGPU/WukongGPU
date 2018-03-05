#pragma once

#include <cuda_runtime.h>
#include "config.hpp"
#include "unit.hpp"

// implement a GPU memory allocator
enum gpu_mem_type {
    RDMA_MEM,
    THREAD_MEM,
    SHARD_MEM,
    NGPU_MEM_TYPE
};


// Represents a GPU memory buffer, its usage is specified by type
struct gpu_mem_t {
    int devid;
    enum gpu_mem_type type;
	char *buf;
    uint64_t size;
    gpu_mem_t *prev;
    gpu_mem_t *next;

    gpu_mem_t(int dev, char *buf, enum gpu_mem_type type, uint64_t sz)
        : devid(dev), buf(buf), type(type), size(sz)
    { prev = next = NULL; }
};


// Each GPU has its own allocator
class GPU_Allocator {
private:
    struct dlist {
        gpu_mem_t *head;
        gpu_mem_t *tail;
        dlist() : head(0), tail(0) {
            // empty
        }
    };

    int devid;
    int num_servers;
    int num_threads;
    int num_shards;
    dlist free_lists[NGPU_MEM_TYPE];
    char *mem;

    void put_to_list(dlist &list, gpu_mem_t *chunk) {
        if (chunk == nullptr)
            return;

        if (list.head == 0 && list.tail == 0) {
            list.head = chunk;
            list.tail = chunk;
        } else {
            list.tail->next = chunk;
            chunk->prev = list.tail;
            list.tail = chunk;
        }
    }

    GPU_Allocator(int devid, int num_servers, int num_threads, int num_shards)
        : devid(devid), num_servers(num_servers), num_threads(num_threads),
          num_shards(num_shards) {

        size_t free_sz, total_sz;
        uint64_t buf_sz, buf_off, rbf_sz, rdma_sz, thbf_sz, shard_sz;
        char *gbuf, *ptr;

        buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
        rbf_sz = MiB2B(global_gpu_rdma_rbf_size_mb);
        thbf_sz = MiB2B(global_gpu_thread_buf_size_mb);
        shard_sz = MiB2B(global_gpu_shard_size_mb);

        // calculate RDMA buffer size
        rdma_sz = buf_sz * num_threads + rbf_sz * num_servers * num_threads;
        cout << "GPU_Allocator: devid=" << devid << endl;

        GPU_ASSERT( cudaSetDevice(devid) );
        GPU_ASSERT( cudaMemGetInfo(&free_sz, &total_sz) );

        GPU_ASSERT( cudaMalloc(&gbuf, rdma_sz) );
        GPU_ASSERT( cudaMemset(gbuf, 0, rdma_sz) );

        put_to_list(free_lists[RDMA_MEM], new gpu_mem_t(devid, gbuf, RDMA_MEM, rdma_sz));

        GPU_ASSERT( cudaMemGetInfo(&free_sz, &total_sz) );

        // prealloc all GPU memory
/*         GPU_ASSERT( cudaMalloc(&mem, free_sz) );
 *         ptr = mem;
 * 
 *         // split the whole buffer to chunks
 *         for (int i = 0; i < num_threads; i++) {
 *             put_to_list(free_lists[THREAD_MEM], new gpu_mem_t(devid, ptr, THREAD_MEM, thbf_sz));
 *             ptr += thbf_sz;
 *         }
 * 
 *         for (int i = 0; i < num_shards; i++) {
 *             put_to_list(free_lists[SHARD_MEM], new gpu_mem_t(devid, ptr, SHARD_MEM, shard_sz));
 *             ptr += shard_sz;
 *         } */

    }

public:
    static GPU_Allocator& get_instance(int devid) {
        static GPU_Allocator instance(devid, global_num_servers, global_num_gpu_engines, global_num_gpu_shards);
        return instance;
    }

    gpu_mem_t *alloc(enum gpu_mem_type type) {
        if (type >= NGPU_MEM_TYPE || !free_lists[type].head)
            return nullptr;

        gpu_mem_t *chunk;
        // remove a chunk from tail, then return it
        chunk = free_lists[type].tail;
        free_lists[type].tail = chunk->prev;
        if (free_lists[type].tail == nullptr)
            free_lists[type].head = nullptr;

        chunk->next = chunk->prev = nullptr;
        return chunk;
    }

    void free(gpu_mem_t *gmem) {
        if (gmem == nullptr)
            return;
        // put to free list
        put_to_list(free_lists[gmem->type], gmem);
    }

    ~GPU_Allocator() {
        gpu_mem_t *gmem;
        // free managed GPU memory
        gmem = free_lists[RDMA_MEM].head;
        if (gmem) {
            GPU_ASSERT( cudaFree(gmem->buf) );
        }

        // free not managed GPU memory
        if (mem)
            GPU_ASSERT( cudaFree(mem) );
    }

};

