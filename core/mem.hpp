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

#include "unit.hpp"
#include "defines.hpp"

extern int global_memstore_size_gb;
extern int global_rdma_buf_size_mb;
extern int global_rdma_rbf_size_mb;

using namespace std;


struct proxy_sentry {
    int flag;
};


class Mem {
private:
	int num_servers;
	int num_threads;

    proxy_sentry *sentry_ptr;
    uint64_t sentry_off;


	// The Wukong's memory layout: kvstore | rdma-buffer | ring-buffer
	// The rdma-buffer and ring-buffer are only used when HAS_RDMA
	char *mem;
	uint64_t mem_sz;

	char *kvs;
	uint64_t kvs_sz;
	uint64_t kvs_off;

#ifdef HAS_RDMA
	char *buf; // #threads
	uint64_t buf_sz;
	uint64_t buf_off;

	char *rbf; // #thread x #servers
	uint64_t rbf_sz;
	uint64_t rbf_off;
#endif

    char *heads;
    uint64_t heads_sz;
    uint64_t heads_off;

    char *remote_heads;     //for rdma to store heads fectched from other servers
    uint64_t remote_heads_off;

public:
	Mem(int num_servers, int num_threads)
		: num_servers(num_servers), num_threads(num_threads) {

		// calculate memory usage
		kvs_sz = GiB2B(global_memstore_size_gb);
#ifdef HAS_RDMA
		buf_sz = MiB2B(global_rdma_buf_size_mb);
		rbf_sz = MiB2B(global_rdma_rbf_size_mb);
#endif

        heads_sz = num_servers * num_threads * sizeof(uint64_t);

#ifdef HAS_RDMA
		// mem_sz = kvs_sz + buf_sz * num_threads + rbf_sz * num_servers * num_threads;
        mem_sz = kvs_sz + buf_sz * num_threads + rbf_sz * num_servers * num_threads + heads_sz * 2;
#else
		mem_sz = kvs_sz;
#endif
        // Siyuan: 用pinned memory试试
        mem = (char *)malloc(mem_sz);
		memset(mem, 0, mem_sz);

		kvs_off = 0;
		kvs = mem + kvs_off;
#ifdef HAS_RDMA
		buf_off = kvs_off + kvs_sz;
		buf = mem + buf_off;
		rbf_off = buf_off + buf_sz * num_threads;
		rbf = mem + rbf_off;
#endif

        heads_off = rbf_off + rbf_sz * num_servers * num_threads;
        assert(heads_off < mem_sz);
        heads = mem + heads_off;

        remote_heads_off = heads_off + heads_sz;
        assert(remote_heads_off < mem_sz);
        remote_heads =  mem + remote_heads_off;

        sentry_off = kvs_sz - sizeof(proxy_sentry);
        sentry_ptr = (proxy_sentry *)(kvs + sentry_off);
        assert(sentry_ptr->flag == 0);

	}

	~Mem() { free(mem); }

	inline char *memory() { return mem; }
	inline uint64_t memory_size() { return mem_sz; }

	// kvstore
	inline char *kvstore() { return kvs; }
	inline uint64_t kvstore_size() { return kvs_sz; }
	inline uint64_t kvstore_offset() { return kvs_off; }

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

    // head of each ring-buffer
    inline char *head(int tid, int dst_sid) { return heads + (tid * num_servers + dst_sid) * sizeof(uint64_t); }
    inline uint64_t head_offset(int tid, int dst_sid) { return heads_off + (tid * num_servers + dst_sid) * sizeof(uint64_t); }

    inline char *remote_head(int dst_sid, int dst_tid) { return remote_heads + (dst_sid * num_threads + dst_tid) * sizeof(uint64_t); }
    inline uint64_t remote_head_offset(int dst_sid, int dst_tid) { return remote_heads_off + (dst_sid * num_threads + dst_tid) * sizeof(uint64_t); }

    inline struct proxy_sentry *sentry() { return sentry_ptr; }
    inline uint64_t sentry_offset() { return sentry_off; }



}; // end of class Mem
