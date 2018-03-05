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
#include "rdma_resource.hpp"
#include "mem.hpp"
#include "gpu_mem.hpp"
#include "taskq_meta.hpp"
#include <string>
#include <cstdio>
#include <cassert>


using namespace std;

extern bool global_disable_gpudirect;

class GDR_Transport {

private:
    GPUMem *gmem;
    Mem *mem;
    int sid;
    int num_servers;
    int num_threads;

    // bool check(int tid, int dst_sid);

    // read data from ring buffer to GPU memory buffer
    // uint64_t fetch(int tid, int dst_sid, char *gpu_buf, uint64_t buf_sz);

public:
    GDR_Transport(int sid, GPUMem *gmem, Mem *mem, int num_servers, int num_threads)
        : sid(sid), gmem(gmem), mem(mem), num_servers(num_servers), num_threads(num_threads) { }

    ~GDR_Transport() { }

    Mem *get_mem() { return mem; }

    void send(int tid, int dst_sid, int dst_tid, const char *data, uint64_t data_sz, rdma_mem_t rdma_mem) {
        if (global_disable_gpudirect)
            assert(false);

        assert(rdma_mem.dst == CPU_DRAM);

        // TODO: GPU engine和engine能够共用一个rmetas吗？
        rbf_rmeta_t *rmeta = TaskQ_Meta::get_remote(dst_sid, dst_tid);
        assert(rdma_mem.dst == CPU_DRAM);
        // uint64_t rbf_sz = (rdma_mem.dst == GPU_DRAM ? gmem->ring_size() : mem->ring_size());
        uint64_t rbf_sz = mem->ring_size();

        uint64_t off = rdma_mem.remote_off;
        // msg: header + data + footer (use data_sz as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        assert(msg_sz < rbf_sz);

        // must send to remote host
        assert(sid != dst_sid);

        uint64_t *hdr_ptr = nullptr, *ftr_ptr = nullptr;
        uint64_t hdr_value, ftr_value;

        // prepare RDMA buffer for RDMA-WRITE
        char *rdma_buf = gmem->buffer(tid);
        GPU_ASSERT( cudaMemcpy(rdma_buf, &data_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) ); // header

        hdr_ptr = (uint64_t *)rdma_buf;

        rdma_buf += sizeof(uint64_t);
        GPU_ASSERT( cudaMemcpy(rdma_buf, data, data_sz, cudaMemcpyDeviceToDevice) );    // data

        rdma_buf += ceil(data_sz, sizeof(uint64_t));
        GPU_ASSERT( cudaMemcpy(rdma_buf, &data_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) );  // footer

        ftr_ptr = (uint64_t*)rdma_buf;

        // write msg to the remote physical-queue
        RDMA &rdma = RDMA::get_rdma();
        // uint64_t rdma_off = (rdma_mem.dst == GPU_DRAM ? gmem->ring_offset(dst_tid, sid) : mem->ring_offset(dst_tid, sid));
        uint64_t rdma_off = mem->ring_offset(dst_tid, sid);

        if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz), rdma_mem);
        } else {
            uint64_t _sz = rbf_sz - (off % rbf_sz);
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid), _sz, rdma_off + (off % rbf_sz), rdma_mem);
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid) + _sz, msg_sz - _sz, rdma_off, rdma_mem);
        }

#ifdef GDR_DEBUG
        if (dst_sid == 1) {
            GPU_ASSERT( cudaMemcpy(&hdr_value, hdr_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // header
            GPU_ASSERT( cudaMemcpy(&ftr_value, ftr_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost) ); // footer

            printf("[DEBUG#%d] GDR send msg[hdr:%lu, ftr:%lu] to server %d\n", sid, hdr_value, ftr_value, dst_sid);
        }

#endif

    }
    // uint64_t recv(int tid, int dst_sid, char *gpu_buf, uint64_t buf_sz);
    // bool tryrecv(int tid, char *gpu_buf, uint64_t buf_sz, uint64_t &data_sz);

};
