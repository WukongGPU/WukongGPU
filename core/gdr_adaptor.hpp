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

#include <string>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <errno.h>
#include <sstream>

#include "config.hpp"
#include "rdma_resource.hpp"
#include "gpu_mem.hpp"

using namespace std;

#define WK_CLINE 64

// The communication over GPU RDMA-based ring buffer
class GDR_Adaptor {
private:
    GPUMem *mem;    // use unified memory save a lot of work
    int sid;
    int num_servers;
    int num_threads;

    // the ring-buffer space contains #threads logical-queues.
    // each logical-queue contains #servers physical queues (ring-buffer).
    // the X physical-queue (ring-buffer) is written by the responding threads
    // (proxies and engine with the same "tid") on the X server.
    //
    // access mode of physical queue is N writers (from the same server) and 1 reader.
    struct rbf_rmeta_t {
        uint64_t tail; // write from here
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    struct rbf_lmeta_t {
        uint64_t head; // read from here
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    rbf_rmeta_t *rmetas;
    rbf_lmeta_t *lmetas;

    // each thread uses a round-robin strategy to check its physical-queues
    struct scheduler_t {
        uint64_t rr_cnt; // round-robin
    } __attribute__ ((aligned (WK_CLINE)));

    scheduler_t *schedulers;

    uint64_t inline floor(uint64_t original, uint64_t n) {
        assert(n != 0);
        return original - original % n;
    }

    uint64_t inline ceil(uint64_t original, uint64_t n) {
        assert(n != 0);
        if (original % n == 0)
            return original;
        return original - original % n + n;
    }

    bool check(int tid, int dst_sid) {
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        char *rbf = mem->ring(tid, dst_sid);
        uint64_t rbf_sz = mem->ring_size();
        // volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header
        union {
            uint8_t data[8];
            uint64_t data_sz;
        } u;
        GPU_ASSERT( cudaMemcpy(u.data, (rbf + lmeta->head % rbf_sz), sizeof(uint64_t), cudaMemcpyDeviceToHost) );

        return (u.data_sz != 0);
    }

    // Siyuan: read data from ring buffer to GPU memory buffer
    uint64_t fetch(int tid, int dst_sid, char *gpu_buf, uint64_t buf_sz) {
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        char * rbf = mem->ring(tid, dst_sid);   // ring(1,0)
        uint64_t rbf_sz = mem->ring_size();
        //volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header
        union {
            uint8_t data[8];
            uint64_t data_sz;
        } u;
        GPU_ASSERT( cudaMemcpy(u.data, (rbf + lmeta->head % rbf_sz), sizeof(uint64_t), cudaMemcpyDeviceToHost) );
        uint64_t data_sz = u.data_sz;

        uint64_t t1 = timer::get_usec();

        //*(uint64_t *)(rbf + lmeta->head % rbf_sz) = 0;  // clean header
        GPU_ASSERT( cudaMemset(rbf + lmeta->head % rbf_sz, 0, sizeof(uint64_t)) );

        uint64_t to_footer = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        volatile uint64_t * footer = (volatile uint64_t *)(rbf + (lmeta->head + to_footer) % rbf_sz); // footer

        GPU_ASSERT( cudaMemcpy(u.data, footer, sizeof(uint64_t), cudaMemcpyDeviceToHost) );
        // GPU_ASSERT( cudaDeviceSynchronize() );

        while (u.data_sz != data_sz) { // spin-wait RDMA-WRITE done
            _mm_pause();
            //assert(*footer == 0 || *footer == data_sz);
            // Siyuan: LUBM-10240 3node的q1和q7这一行assert会报错
            assert(u.data_sz == 0 || u.data_sz == data_sz);
            GPU_ASSERT( cudaMemcpy(u.data, footer, sizeof(uint64_t), cudaMemcpyDeviceToHost) );
        }
        //*footer = 0;  // clean footer
        GPU_ASSERT( cudaMemset(footer, 0, sizeof(uint64_t)) );

        uint64_t t2 = timer::get_usec();

        // read data
        /* std::string result;
         * result.reserve(data_sz); */
        uint64_t start = (lmeta->head + sizeof(uint64_t)) % rbf_sz;
        uint64_t end = (lmeta->head + sizeof(uint64_t) + data_sz) % rbf_sz;
        if (start < end) {
            /* result.append(rbf + start, data_sz); */
            assert(data_sz < buf_sz);

            /* memcpy(gpu_buf, rbf + start, data_sz); */
            GPU_ASSERT( cudaMemcpy(gpu_buf, rbf + start, data_sz, cudaMemcpyDeviceToDevice) );

            // clear the slot in ring buffer
            //memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));
            GPU_ASSERT( cudaMemset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t))) );  // clean data
        } else {
            /* result.append(rbf + start, data_sz - end);
             * result.append(rbf, end); */
            /* memcpy(gpu_buf, rbf + start, data_sz - end);
             * memcpy(gpu_buf + (data_sz - end), rbf, end); */
            GPU_ASSERT( cudaMemcpy(gpu_buf, rbf + start, data_sz - end, cudaMemcpyDeviceToDevice) );
            GPU_ASSERT( cudaMemcpy(gpu_buf + (data_sz - end), rbf, end, cudaMemcpyDeviceToDevice) );

            GPU_ASSERT( cudaMemset(rbf + start, 0, data_sz - end) );                    // clean data
            //memset(rbf + start, 0, data_sz - end);
            GPU_ASSERT( cudaMemset(rbf, 0, ceil(end, sizeof(uint64_t))) );              // clean data
            //memset(rbf, 0, ceil(end, sizeof(uint64_t)));
        }
        lmeta->head += 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));

#ifdef WUKONG_DEBUG
        printf("[INFO#%d] gdr_adaptor():fetch() recv %llu bytes\n", sid, data_sz);
#endif

        uint64_t t3 = timer::get_usec();

        return data_sz;
    }

public:
    GDR_Adaptor(int sid, GPUMem *mem, int num_servers, int num_threads)
        : sid(sid), mem(mem), num_servers(num_servers), num_threads(num_threads) {

        // init the metadata of remote and local ring-buffers
        int nrbfs = num_servers * num_threads;

        rmetas = (rbf_rmeta_t *)malloc(sizeof(rbf_rmeta_t) * nrbfs);
        memset(rmetas, 0, sizeof(rbf_rmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            rmetas[i].tail = 0;
            pthread_spin_init(&rmetas[i].lock, 0);
        }

        lmetas = (rbf_lmeta_t *)malloc(sizeof(rbf_lmeta_t) * nrbfs);
        memset(lmetas, 0, sizeof(rbf_lmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            lmetas[i].head = 0;
            pthread_spin_init(&lmetas[i].lock, 0);
        }

        schedulers = (scheduler_t *)malloc(sizeof(scheduler_t) * num_threads);
        memset(schedulers, 0, sizeof(scheduler_t) * num_threads);
    }

    ~GDR_Adaptor() { }

    /**
     *
     * @data: pointer on GPU
     */
    void send(int tid, int dst_sid, int dst_tid, const char *data, uint64_t data_sz) {
        assert(tid == dst_tid);
#ifdef WUKONG_DEBUG
        printf("[INFO#%d] gdr_adaptor:send(): dst_sid=%d, dst_tid=%d, data_sz=%llu\n", sid, dst_sid, dst_tid, data_sz);
#endif
        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];

        uint64_t rbf_sz = mem->ring_size();

        union {
            uint8_t data[8];
            uint64_t data_sz;
        } u;
        u.data_sz = data_sz;

        // msg: header + data + footer (use data_sz as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        assert(msg_sz < rbf_sz);
        /// TODO: check overwriting (i.e., (tail + msg_sz) % rbf_sz >= head)
        /// maintain a stale header for each remote ring buffer, and update it when may occur overwriting

        pthread_spin_lock(&rmeta->lock);

        if (sid == dst_sid) { // local physical-queue
            uint64_t off = rmeta->tail;
            rmeta->tail += msg_sz;

            pthread_spin_unlock(&rmeta->lock);

            // write msg to the local physical-queue
            char *ptr = mem->ring(dst_tid, sid);    // ring(1,0)
            GPU_ASSERT( cudaMemcpy(ptr + off % rbf_sz, u.data, sizeof(uint64_t), cudaMemcpyHostToDevice) ); // header
            off += sizeof(uint64_t);
            if (off / rbf_sz == (off + data_sz - 1) / rbf_sz ) { // data
                GPU_ASSERT( cudaMemcpy(ptr + (off % rbf_sz), data, data_sz, cudaMemcpyDeviceToDevice) );
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                GPU_ASSERT( cudaMemcpy(ptr + (off % rbf_sz), data, _sz, cudaMemcpyDeviceToDevice) );
                GPU_ASSERT( cudaMemcpy(ptr, data + _sz, data_sz - _sz, cudaMemcpyDeviceToDevice) );
            }
            off += ceil(data_sz, sizeof(uint64_t));
            GPU_ASSERT( cudaMemcpy(ptr + off % rbf_sz, u.data, sizeof(uint64_t), cudaMemcpyHostToDevice) ); // footer

        } else { // remote physical-queue
            uint64_t off = rmeta->tail;
            rmeta->tail += msg_sz;
            pthread_spin_unlock(&rmeta->lock);

            pthread_spin_unlock(&rmeta->lock);

            // prepare RDMA buffer for RDMA-WRITE
            char *rdma_buf = mem->buffer(tid);
            GPU_ASSERT( cudaMemcpy(rdma_buf, u.data, sizeof(uint64_t), cudaMemcpyHostToDevice) ); // header

            rdma_buf += sizeof(uint64_t);
            assert((int64_t)(rdma_buf - mem->buffer(tid)) < (int64_t)mem->buffer_size());
            assert((int64_t)(rdma_buf + data_sz - mem->buffer(tid)) <  (int64_t)mem->buffer_size());
            // printf("GDR_Adaptor: rdma_buf=%p, data=%p, data_sz=%llu, buf_sz=%llu\n", rdma_buf, data, data_sz, mem->buffer_size());
            GPU_ASSERT( cudaMemcpy(rdma_buf, data, data_sz, cudaMemcpyDeviceToDevice) );    // data

            rdma_buf += ceil(data_sz, sizeof(uint64_t));
            GPU_ASSERT( cudaMemcpy(rdma_buf, u.data, sizeof(uint64_t), cudaMemcpyHostToDevice) );  // footer

            // write msg to the remote physical-queue
            RDMA &rdma = RDMA::get_rdma();
            uint64_t rdma_off = mem->ring_offset(dst_tid, sid);
            if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
                rdma.dev->GPURdmaWrite(tid, dst_sid, mem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz));
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                rdma.dev->GPURdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
                rdma.dev->GPURdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, msg_sz - _sz, rdma_off);
            }
        }
    }

    /**
     *
     * @return: number of bytes received
     */
    uint64_t recv(int tid, int dst_sid, char *gpu_buf, uint64_t buf_sz) {
        while (true) {
            // each thread has a logical-queue (#servers physical-queues)
            // now we specify recv destination instead of rr
            //int dst_sid = (schedulers[tid].rr_cnt++) % num_servers; // round-robin
            if (check(tid, dst_sid))
                return fetch(tid, dst_sid, gpu_buf, buf_sz);
        }
    }

    bool tryrecv(int tid, char *gpu_buf, uint64_t buf_sz, uint64_t &data_sz) {
        // check all physical-queues once
        for (int dst_sid = 0; dst_sid < num_servers; dst_sid++) {
            if (check(tid, dst_sid)) {
                data_sz = fetch(tid, dst_sid, gpu_buf, buf_sz);
                return true;
            }
        }
        return false;
    }

};
