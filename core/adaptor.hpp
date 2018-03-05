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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <sstream>

#include "query.hpp"
#include "tcp_transport.hpp"
#include "rdma_transport.hpp"
#include "taskq_meta.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {

    inline bool rbuf_full(int tid, int dst_sid, int dst_tid, uint64_t msg_sz) {
        rbf_rmeta_t *rmeta = TaskQ_Meta::get_remote(dst_sid, dst_tid);//&rmetas[dst_sid * num_threads + dst_tid];
        uint64_t rbf_sz = rdma->get_mem()->ring_size();
        uint64_t head = *(uint64_t *)(rdma->get_mem()->remote_head(dst_sid, dst_tid));

       if (rbf_sz - (rmeta->tail - head) > msg_sz)
               return false;

       return true;
    }

public:
    int tid; // thread id

    TCP_Transport *tcp = NULL;   // communicaiton by TCP/IP
    RDMA_Transport *rdma = NULL; // communicaiton by RDMA


    Adaptor(int tid, TCP_Transport *tcp = NULL, RDMA_Transport *rdma = NULL)
        : tid(tid), tcp(tcp), rdma(rdma) { }

    ~Adaptor() { }


    bool send(int dst_sid, int dst_tid, const request_or_reply &r) {
        assert(r.query_type == FULL_QUERY);
        uint64_t data_sz, msg_sz;
        std::stringstream ss;
        string str;
        boost::archive::binary_oarchive oa(ss);

        oa << r;

        rbf_rmeta_t *rmeta = TaskQ_Meta::get_remote(dst_sid, dst_tid);
        str = ss.str();
        data_sz = str.length();
        msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        if (rdma) {
            // need a lock to protect remote offset, since there are multiple writers
            pthread_spin_lock(&rmeta->lock);

            // Siyuan: 临时注释
            if (rbuf_full(tid, dst_sid, dst_tid, msg_sz)) {
                pthread_spin_unlock(&rmeta->lock);
                return false;
            }
            uint64_t off = rmeta->tail;
            rdma->send(tid, dst_sid, dst_tid, str.c_str(), data_sz, off);
            rmeta->tail += msg_sz;
            pthread_spin_unlock(&rmeta->lock);


        } else {
            cout << "ERORR: attempting to use RDMA adaptor, "
                 << "but Wukong was built without RDMA."
                 << endl;
        }

        return true;
    }

    request_or_reply recv() {
        assert(global_use_rdma);
        std::string str;
        if (rdma) {
            str = rdma->recv(tid);
        } else {
            cout << "ERORR: attempting to use RDMA adaptor, "
                 << "but Wukong was built without RDMA."
                 << endl;
        }

        if (str.empty() || str.length() < 1)
            assert(false);

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        request_or_reply r;
        ia >> r;
        return r;
    }

    bool tryrecv(request_or_reply &r) {
        std::string str;
        if (rdma) {
            int dst_sid_out = 0;
            if (!rdma->tryrecv(tid, dst_sid_out, str))
                return false;
        } else {
            cout << "ERORR: attempting to use RDMA adaptor, "
                 << "but Wukong was built without RDMA."
                 << endl;
        }

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        ia >> r;
        return true;
    }

};
