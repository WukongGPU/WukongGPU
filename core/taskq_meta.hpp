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
#include <stdlib.h>
#include <unistd.h>

#include "defines.hpp"

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

// each thread uses a round-robin strategy to check its physical-queues
struct scheduler_t {
    uint64_t rr_cnt; // round-robin
} __attribute__ ((aligned (WK_CLINE)));


class TaskQ_Meta {
private:

    static rbf_rmeta_t *rmetas;
    static rbf_lmeta_t *lmetas;
    static int num_servers, num_threads;

public:
    static void init(int nservers, int nthreads) {
        // init the metadata of remote and local ring-buffers
        int nrbfs = nservers * nthreads;
        TaskQ_Meta::num_servers = nservers;
        TaskQ_Meta::num_threads = nthreads;

        TaskQ_Meta::rmetas = (rbf_rmeta_t *)malloc(sizeof(rbf_rmeta_t) * nrbfs);
        memset(rmetas, 0, sizeof(rbf_rmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            rmetas[i].tail = 0;
            pthread_spin_init(&rmetas[i].lock, 0);
        }

        TaskQ_Meta::lmetas = (rbf_lmeta_t *)malloc(sizeof(rbf_lmeta_t) * nrbfs);
        memset(lmetas, 0, sizeof(rbf_lmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            lmetas[i].head = 0;
            pthread_spin_init(&lmetas[i].lock, 0);
        }
    }

    static rbf_lmeta_t *get_local(int tid, int dst_sid) {
        assert(tid < num_threads);
        return &(TaskQ_Meta::lmetas[tid * num_servers + dst_sid]);
    }

    static rbf_rmeta_t *get_remote(int dst_sid, int dst_tid) {
        assert(dst_tid < num_threads);
        return &(TaskQ_Meta::rmetas[dst_sid * num_threads + dst_tid]);
    }
};

/* initialize static members */
rbf_rmeta_t* TaskQ_Meta::rmetas = nullptr;
rbf_lmeta_t* TaskQ_Meta::lmetas = nullptr;
int TaskQ_Meta::num_servers = global_num_servers;
int TaskQ_Meta::num_threads = global_num_threads;

