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
 */

#pragma once

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "config.hpp"
#include "query.hpp"
#include "tcp_adaptor.hpp"
#include "gdr_adaptor.hpp"
#include "rdma_adaptor.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {
public:
    int tid; // thread id
    int gpu_tid;    // for GPUDirect RDMA

    TCP_Adaptor *tcp = NULL;   // communicaiton by TCP/IP
    RDMA_Adaptor *rdma = NULL; // communicaiton by RDMA
    GDR_Adaptor *gdr = NULL;

    Adaptor(int tid, TCP_Adaptor *tcp = NULL, RDMA_Adaptor *rdma = NULL, GDR_Adaptor *gdr = NULL)
        : tid(tid), tcp(tcp), rdma(rdma), gdr(gdr) {
            gpu_tid = tid - (global_num_proxies + global_num_engines);
        }

    ~Adaptor() { }

    void send(int dst_sid, int dst_tid, request_or_reply &r) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        if (global_use_rdma) {
            if (rdma) {
                rdma->send(tid, dst_sid, dst_tid, ss.str());
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            tcp->send(dst_sid, dst_tid, ss.str());
        }
    }

    void gpu_send(int dst_sid, request_or_reply &r) {
        assert(gpu_tid >= 0);
        assert(gpu_tid < global_num_gpu_engines);

        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        // send history table
        gdr->send(gpu_tid, dst_sid, gpu_tid, r.gpu_history_table_ptr, r.gpu_history_table_size*sizeof(int));

        oa << r;
        // send request
        rdma->send(tid, dst_sid, tid, ss.str());
    }

    //no suitable usage scenario
    /*request_or_reply gpu_recv(char *gpu_buf, uint64_t buf_sz) {*/
        //// receive history table first
        //uint64_t data_sz = 0;
        //if (gdr) {
            //data_sz = gdr->recv(gpu_tid, gpu_buf, buf_sz);
        //} else {
            //cout << "ERORR: attempting to use GDR adaptor, "
                 //<< "but Wukong was built without GDR."
                 //<< endl;
        //}

        //// receive query request later
        //std::string str;
        //str = rdma->recv(tid);
        //std::stringstream ss;
        //ss << str;

        //boost::archive::binary_iarchive ia(ss);
        //request_or_reply r;
        //ia >> r;

        //r.gpu_history_table_ptr = gpu_buf;
        //r.gpu_history_table_size = data_sz/sizeof(int);

        //return r;
    /*}*/

    request_or_reply recv() {
        std::string str;
        if (global_use_rdma) {
            if (rdma) {
                str = rdma->recv(tid);
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            str = tcp->recv(tid);
        }

        if (str.empty() || str.length() < 1)
            return request_or_reply(true);

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        request_or_reply r;
        ia >> r;
        return r;
    }

    bool gpu_tryrecv(request_or_reply &r, char *gpu_buf, uint64_t buf_sz) {
        assert(gpu_tid >= 0);

        std::string str;
        int dst_sid_out = 0;
        if (!rdma->tryrecv(tid,dst_sid_out, str))
            return false;

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        ia >> r;

        uint64_t data_sz = 0;
        if (gdr) {
            if (!r.is_request() || r.gpu_history_table_size == 0)
                return true;
            else
                data_sz = gdr->recv(gpu_tid, dst_sid_out, gpu_buf, buf_sz);
        } else {
            cout << "ERORR: attempting to use GDR adaptor, "
                 << "but Wukong was built without GDR."
                 << endl;
        }

        r.gpu_history_table_ptr = gpu_buf;
        r.gpu_history_table_size = data_sz/sizeof(int);
        return true;
    }

    bool tryrecv(request_or_reply &r) {
        std::string str;
        if (global_use_rdma) {
            if (rdma) {
                int dst_sid_out = 0;
                if (!rdma->tryrecv(tid, dst_sid_out, str)) return false;
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            if (!tcp->tryrecv(tid, str)) return false;
        }

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        ia >> r;
        return true;
    }

};
