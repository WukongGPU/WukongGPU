/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 *
 */

#pragma once

#include "gpu_malloc.hpp"
#include "gpu_mem.hpp"
#include "unit.hpp"


class GPUMem {
private:
    int devid;
	int num_servers;
	int num_agents;
    gpu_mem_t *gmem;

	// The GPU memory layout: sharding kvstore | history | rdma-buffer | heap
	char *mem;
	uint64_t mem_sz;

	// char *kvs;
	// uint64_t kvs_sz;
	// uint64_t kvs_off;

	char *buf; // #threads
	uint64_t buf_sz;
	uint64_t buf_off;

public:
    GPUMem(int devid, int num_servers, int num_agents)
        : devid(devid), num_servers(num_servers), num_agents(num_agents) {

        // calculate memory usage
        // kvs_sz = GiB2B(global_memstore_size_gb);
        buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
        // rbf_sz = MiB2B(global_gpu_rdma_rbf_size_mb);

        // mem_sz = kvs_sz + buf_sz * num_agents + rbf_sz * num_servers * num_agents;
        mem_sz = buf_sz * num_agents;

        gmem = GPU_Allocator::get_instance(devid).alloc(RDMA_MEM);
        mem = gmem->buf;
        assert(mem_sz <= gmem->size);

        buf_off = 0;
        buf = mem + buf_off;

        printf("[INFO] GPUMem: num_servers=%d, num_agents=%d\n", num_servers, num_agents);

    }

    ~GPUMem() { GPU_Allocator::get_instance(devid).free(gmem); }

	inline char *memory() { return mem; }
	inline uint64_t memory_size() { return mem_sz; }

	// kvstore
	// inline char *kvstore() { return kvs; }
	// inline uint64_t kvstore_size() { return kvs_sz; }
	// inline uint64_t kvstore_offset() { return kvs_off; }

	// buffer
	inline char *buffer(int tid) { return buf + buf_sz * (tid % num_agents); }
	inline uint64_t buffer_size() { return buf_sz; }
	inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }

	// ring-buffer
    // inline char *ring(int tid, int sid) { assert(false); }
    // inline uint64_t ring_size() { assert(false); }
    // inline uint64_t ring_offset(int tid, int sid) { assert(false); }

}; // end of class GPUMem
