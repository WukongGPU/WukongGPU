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
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <cassert>

#include <string>
#include "rdma_resource.hpp"
#include "taskq_meta.hpp"
#include "mem.hpp"
#include "defines.hpp"
#include "gpu.hpp"

// The communication over RDMA-based ring buffer
class RDMA_Transport {
private:
    Mem *mem;
    int sid;
    int num_servers;
    int num_threads;

    scheduler_t *schedulers;

    void update_rbuf_head(int tid, int dst_sid, rbf_lmeta_t *lmeta) {
        uint64_t rbf_sz = mem->ring_size();
        /* code for overwrite detection */
        char *head = mem->head(tid, dst_sid);
        // TODO: 这个threshold要看看是不是导致更新频率太低了
        if(lmeta->head - *(uint64_t *)head > rbf_sz / 32) {
            *(uint64_t *)head = lmeta->head;
            if(sid != dst_sid){
                RDMA &rdma = RDMA::get_rdma();
                uint64_t remote_head = mem->remote_head_offset(sid, tid);
                rdma.dev->RdmaWrite(tid, dst_sid, head, sizeof(uint64_t), remote_head);
            }
            else{
                *(uint64_t *)mem->remote_head(sid, tid) = lmeta->head;
            }
        }
    }

    bool check(int tid, int dst_sid) {
        rbf_lmeta_t *lmeta = TaskQ_Meta::get_local(tid, dst_sid);//&lmetas[tid * num_servers + dst_sid];
        char *rbf = mem->ring(tid, dst_sid);
        uint64_t rbf_sz = mem->ring_size();
        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header

        return (data_sz != 0);
    }

    // fetch message from CPU ring buffer
    bool fetch(int tid, int dst_sid, std::string &result, enum MemTypes memtype) {
        rbf_lmeta_t *lmeta = TaskQ_Meta::get_local(tid, dst_sid);//&lmetas[tid * num_servers + dst_sid];
        char * rbf = mem->ring(tid, dst_sid);
        uint64_t rbf_sz = mem->ring_size();

        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header

        uint64_t t1 = timer::get_usec();

        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        uint64_t to_footer = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        uint64_t footer_off = (lmeta->head + to_footer) % rbf_sz;

#ifdef GDR_DEBUG
        uint32_t *myptr;
        char *msg_head = (rbf + lmeta->head % rbf_sz);
        // debug code
        if (sid == 1 && dst_sid == 0 && memtype == GPU_DRAM) {
            printf("[DEBUG#%d] RDMA begin recv msg[hdr:%lu, ftr:xxx] from server %d, local_head: %lu, footer_off: %lu.\n",
                    sid, data_sz, dst_sid, lmeta->head, footer_off);
        // }

        // if (sid == 1 && dst_sid == 0) {
            if (data_sz == 3640) {
                printf("[DEBUG#%d] receiver dump msg from server %d:\n", sid, dst_sid);
                myptr = (uint32_t *)msg_head;  //(rbf + (lmeta->head + sizeof(uint64_t)) % rbf_sz);
                for (int i = 0; i < msg_sz / sizeof(uint32_t); ++i) {
                    printf("0x%x ", myptr[i]);
                    if ((i+1) % 8 == 0) {
                        printf("\n");
                    }
                }
                printf("\n================ receiver#%d dump msg [data_sz: %lu] =================\n", sid, data_sz);
            }
        }
#endif

        *(uint64_t *)(rbf + lmeta->head % rbf_sz) = 0;  // clean header
        // *(uint64_t *)(msg_head) = 0;  // clean header
        volatile uint32_t *last_word = (uint32_t *)(rbf + (lmeta->head + sizeof(uint64_t) + data_sz) % rbf_sz) - 1;
        volatile uint64_t *footer = (volatile uint64_t *)(rbf + footer_off); // footer

        while (*footer != data_sz) { // spin-wait RDMA-WRITE done
            _mm_pause();
            // assert(*footer == 0 || *footer == data_sz);
            if (memtype == GPU_DRAM && *last_word != 0)
                break;
        }

#ifdef GDR_DEBUG
        if (sid == 1 && dst_sid == 0) {
            printf("[DEBUG#%d] RDMA finish recv msg[hdr:%lu, ftr:%lu] from server %d. local_head: %lu\n",
                    sid, data_sz, *footer, dst_sid, lmeta->head);
        }
#endif

        *footer = 0;  // clean footer

        uint64_t start = (lmeta->head + sizeof(uint64_t)) % rbf_sz;
        uint64_t end = (lmeta->head + sizeof(uint64_t) + data_sz) % rbf_sz;

        // TODO: directly copy data to GPU
        if (memtype == GPU_DRAM) {
            GPU &gpu = GPU::instance();
            char *history_buf = gpu.history_inbuf();
            if (start < end) {
                GPU_ASSERT( cudaMemcpy(history_buf, rbf + start, data_sz, cudaMemcpyHostToDevice) );
                memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
            } else {
                GPU_ASSERT( cudaMemcpy(history_buf, rbf + start, data_sz - end, cudaMemcpyHostToDevice) );
                GPU_ASSERT( cudaMemcpy(history_buf + (data_sz - end), rbf, end, cudaMemcpyHostToDevice) );
                memset(rbf + start, 0, data_sz - end);                    // clean data
                memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
            }
            gpu.set_history_size(data_sz / sizeof(int));
            // TODO: 把data拷贝上去之后，caller还需要设置r.gpu_history_ptr, gpu_history_size这些源数据

        } else {
            // read data
            result.reserve(data_sz);
            if (start < end) {
                result.append(rbf + start, data_sz);
                memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
            } else {
                result.append(rbf + start, data_sz - end);
                result.append(rbf, end);
                memset(rbf + start, 0, data_sz - end);                    // clean data
                memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
            }
        }

        // move forward rbf head
        uint64_t delta = ( 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) );

#ifdef GDR_DEBUG
        if (sid == 1 && dst_sid == 0) { 
            printf("[DEBUG#%d] RDMA::fetch() local_head: before: %lu, after: %lu\n", sid, lmeta->head, lmeta->head + delta);
        }
#endif

        lmeta->head += delta;
        update_rbuf_head(tid, dst_sid, lmeta);

        return true;
    }

public:
    RDMA_Transport(int sid, Mem *mem, int num_servers, int num_threads)
        : sid(sid), mem(mem), num_servers(num_servers), num_threads(num_threads) {
        schedulers = (scheduler_t *)malloc(sizeof(scheduler_t) * num_threads);
        memset(schedulers, 0, sizeof(scheduler_t) * num_threads);
    }

    ~RDMA_Transport() { }

    Mem *get_mem() { return mem; }

    // caller may needs to acquire lock, since there wiil be multiple writers
    // void send(int tid, int dst_sid, int dst_tid, const string &str, uint64_t offset) {
    void send(int tid, int dst_sid, int dst_tid, const char *data, uint64_t data_sz, uint64_t offset) {
#ifdef RDMA_DEBUG
        printf("[INFO#%d] RDMA_Transport::send(): dst_sid=%d, dst_tid=%d\n", sid, dst_sid, dst_tid);
#endif

        uint64_t rbf_sz = mem->ring_size();

        // msg: header + data + footer (use data_sz as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        uint64_t off = offset;

        assert(msg_sz < rbf_sz);


        uint64_t *hdr_ptr = nullptr, *ftr_ptr = nullptr;
        if (sid == dst_sid) { // local physical-queue

            // write msg to the local physical-queue
            char *ptr = mem->ring(dst_tid, sid);
            *((uint64_t *)(ptr + off % rbf_sz)) = data_sz;       // header

            hdr_ptr = (uint64_t *)(ptr + off % rbf_sz);

            off += sizeof(uint64_t);
            if (off / rbf_sz == (off + data_sz - 1) / rbf_sz ) { // data
                memcpy(ptr + (off % rbf_sz), data, data_sz);
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                memcpy(ptr + (off % rbf_sz), data, _sz);
                memcpy(ptr, data + _sz, data_sz - _sz);
            }
            off += ceil(data_sz, sizeof(uint64_t));
            *((uint64_t *)(ptr + off % rbf_sz)) = data_sz;       // footer

            ftr_ptr = (uint64_t *)(ptr + off % rbf_sz);

        } else { // remote physical-queue


            // prepare RDMA buffer for RDMA-WRITE
            char *rdma_buf = mem->buffer(tid);
            *((uint64_t *)rdma_buf) = data_sz;  // header

            hdr_ptr = (uint64_t *)rdma_buf;

            rdma_buf += sizeof(uint64_t);
            memcpy(rdma_buf, data, data_sz);    // data
            rdma_buf += ceil(data_sz, sizeof(uint64_t));
            *((uint64_t*)rdma_buf) = data_sz;   // footer

            ftr_ptr = (uint64_t*)rdma_buf;

            // write msg to the remote physical-queue
            RDMA &rdma = RDMA::get_rdma();
            uint64_t rdma_off = mem->ring_offset(dst_tid, sid);
            if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz));
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, msg_sz - _sz, rdma_off);
                printf("[WARN#%d] RDMA abnormal path! remote_off: %lu\n", sid, off);
            }
        }

#ifdef GDR_DEBUG
        if (dst_sid == 1) {
            printf("[DEBUG#%d] RDMA send msg[hdr:%lu, ftr:%lu] to server %d\n", sid, *hdr_ptr, *ftr_ptr, dst_sid);
        }
#endif

        // if (dst_sid == 2) {
            // uint32_t *myptr;
            // // 把sender放在rdma_buf里的message全部dump出来
            // if (data_sz == 3024) {
                // printf("[DEBUG#%d] RDMA sender dump msg_sz: %d\n", sid, msg_sz);
                // myptr = (uint32_t *)(mem->buffer(tid));

                // printf("[DEBUG#%d] >>>> dump data:\n", sid);
                // for (int i = 0; i < msg_sz / sizeof(uint32_t); ++i) {
                    // printf("0x%x ", myptr[i]);
                    // if ((i+1) % 8 == 0) {
                        // printf("\n");
                    // }
                // }
                // printf("\n");
            // }
        // }

    }

    // caller no need to acquire lock, since there is only one reader
    std::string recv(int tid) {
        while (true) {
            // each thread has a logical-queue (#servers physical-queues)
            string str;
            int dst_sid = (schedulers[tid].rr_cnt++) % num_servers; // round-robin
            if (check(tid, dst_sid)) {
                bool ret;
                ret = fetch(tid, dst_sid, str, CPU_DRAM);
                assert(ret == true);
                return str;
            }
        }
    }


    int recv_from(int tid, int dst_sid, std::string &str, enum MemTypes memtype) {
        while (true) {
            if (check(tid, dst_sid)) {
                bool ret;
                ret = fetch(tid, dst_sid, str, memtype);
                assert(ret == true);

                if (memtype == CPU_DRAM)
                    return str.length();
                else
                    return GPU::instance().history_size();
            }
        }
        return -1;
    }

    bool tryrecv(int tid, int &dst_sid_out, std::string &str) {
        // check all physical-queues once
        for (int dst_sid = 0; dst_sid < num_servers; dst_sid++) {
            if (check(tid, dst_sid)) {
                dst_sid_out = dst_sid;
                return fetch(tid, dst_sid, str, CPU_DRAM);
            }
        }
        return false;
    }
};
