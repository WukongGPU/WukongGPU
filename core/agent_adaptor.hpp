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

#include <pthread.h>
#include "taskq_meta.hpp"
#include "adaptor.hpp"
#include "gdr_transport.hpp"
#include "rdma_transport.hpp"

using namespace std;

// Siyuan: 不需要gpu_tid了，因为gpu上没有ring buffer了
class Agent_Adaptor : public Adaptor {

private:
    GDR_Transport *gdr;

    inline bool rbuf_full(int tid, int dst_sid, int dst_tid, uint64_t msg_sz) {
        rbf_rmeta_t *rmeta = TaskQ_Meta::get_remote(dst_sid, dst_tid);//&rmetas[dst_sid * num_threads + dst_tid];
        uint64_t rbf_sz = gdr->get_mem()->ring_size();
        uint64_t head = *(uint64_t *)(gdr->get_mem()->remote_head(dst_sid, dst_tid));

       if (rbf_sz - (rmeta->tail - head) > msg_sz)
               return false;

       return true;
    }

public:
    Agent_Adaptor(int tid, TCP_Transport *tcp = NULL, RDMA_Transport *rdma = NULL, GDR_Transport *gdr = NULL)
        : Adaptor(tid, tcp, rdma), gdr(gdr) {
    }

    // Siyuan: send split query (data on GPU, control on CPU)
    bool send_split(int dst_sid, int dst_tid, const request_or_reply &r, char *history_ptr, uint64_t table_size) {
        assert(tid < global_num_threads);
        assert(r.query_type == SPLIT_QUERY);
        bool success;

        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;

        // data_sz: size of history on GPU
        uint64_t control_msg_sz, data_msg_sz;
        string str = ss.str();

        control_msg_sz = sizeof(uint64_t) + ceil(str.length(), sizeof(uint64_t)) + sizeof(uint64_t);
        data_msg_sz = sizeof(uint64_t) + ceil(table_size * sizeof(int), sizeof(uint64_t)) + sizeof(uint64_t);

        rbf_rmeta_t *rmeta = TaskQ_Meta::get_remote(dst_sid, dst_tid);
        pthread_spin_lock(&rmeta->lock);

        // #0 check whether remote queue is full
        if (rbuf_full(tid, dst_sid, dst_tid, control_msg_sz + data_msg_sz)) {
            pthread_spin_unlock(&rmeta->lock);
            return false;
        }

        // #1 send control object
        rdma->send(tid, dst_sid, dst_tid, str.c_str(), str.length(), rmeta->tail);
        rmeta->tail += control_msg_sz;

        assert(history_ptr != nullptr);
        // #2 send history
        // prepare RDMA parameters
        // w/o GPUDirect, history has been copyied to a host buffer
        if (r.gpu_history_ptr == (char *)0xdeadbeef) {
            rdma->send(tid, dst_sid, dst_tid, history_ptr, table_size * sizeof(int), rmeta->tail);
        } else {
            rdma_mem_t rmem;
            rmem.src = GPU_DRAM;
            rmem.dst = CPU_DRAM;
            rmem.remote_off = rmeta->tail;

            gdr->send(tid, dst_sid, dst_tid, history_ptr, table_size * sizeof(int), rmem);
        }
        rmeta->tail += data_msg_sz;

        pthread_spin_unlock(&rmeta->lock);
        return true;
    }


    // Siyuan: receive的时候不需要acquire lock
    // 因为每一个ring buffer只有一个reader
    bool tryrecv_split(request_or_reply &r) {

        std::string str;
        int sender_sid = 0;

        if (!rdma->tryrecv(tid, sender_sid, str))
            return false;

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        ia >> r;

        // continue receive history of query
        if (r.query_type == SPLIT_QUERY) {
            int ret;
            std::string dumb_str;

            ret = rdma->recv_from(tid, sender_sid, dumb_str, GPU_DRAM);
            assert(ret > 0);
            GPU &gpu = GPU::instance();
            // Siyuan: history已经load上GPU了，但gpu.query_id还没有设置
            r.gpu_history_ptr = gpu.history_inbuf();
            r.gpu_history_table_size = gpu.history_size();
        }
        return true;
    }


};
