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


class GPUMem {
private:
    int devid;
	int num_servers;
	int num_threads;
    gpu_mem_t *gmem;

	// The Wukong's memory layout: kvstore | rdma-buffer | ring-buffer
	// The rdma-buffer and ring-buffer are only used when HAS_RDMA
	char *ptr;
	uint64_t mem_sz;

	/* char *kvs;
	 * uint64_t kvs_sz;
	 * uint64_t kvs_off; */

#ifdef HAS_RDMA
	char *buf; // #threads
	uint64_t buf_sz;
	uint64_t buf_off;

	char *rbf; // #thread x #servers
	uint64_t rbf_sz;
	uint64_t rbf_off;
#endif

public:
	GPUMem(int devid, int num_servers, int num_threads)
		: devid(devid), num_servers(num_servers), num_threads(num_threads) {

		// calculate memory usage
		//kvs_sz = GiB2B(global_memstore_size_gb);
#ifdef HAS_RDMA
		buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
		rbf_sz = MiB2B(global_gpu_rdma_rbf_size_mb);
#endif

#ifdef HAS_RDMA
		// mem_sz = kvs_sz + buf_sz * num_threads + rbf_sz * num_servers * num_threads;
		mem_sz = buf_sz * num_threads + rbf_sz * num_servers * num_threads;
#else
		// mem_sz = kvs_sz;
#endif

		// mem = (char *)malloc(mem_sz);
		// memset(mem, 0, mem_sz);
        gmem = GPU_Allocator::get_instance(devid).alloc(RDMA_MEM);
        ptr = gmem->buf;
        assert(mem_sz <= gmem->size);

#ifdef HAS_RDMA
		buf_off = 0;
		buf = ptr + buf_off;
		rbf_off = buf_off + buf_sz * num_threads;
		rbf = ptr + rbf_off;
#endif

	}

	~GPUMem() { GPU_Allocator::get_instance(devid).free(gmem); }

	inline char *memory() { return ptr; }
	inline uint64_t memory_size() { return mem_sz; }

	// kvstore
	// inline char *kvstore() { return kvs; }
	// inline uint64_t kvstore_size() { return kvs_sz; }
	// inline uint64_t kvstore_offset() { return kvs_off; }

#ifdef HAS_RDMA
	// buffer
	inline char *buffer(int tid) { return buf + buf_sz * tid; }
	inline uint64_t buffer_size() { return buf_sz; }
	inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * tid; }

	// ring-buffer
	inline char *ring(int tid, int sid) { return rbf + (rbf_sz * num_servers) * tid + rbf_sz * sid; }
	inline uint64_t ring_size() { return rbf_sz; }
	inline uint64_t ring_offset(int tid, int sid) { return rbf_off + (rbf_sz * num_servers) * tid + rbf_sz * sid; }
#endif

}; // end of class GPUMem
