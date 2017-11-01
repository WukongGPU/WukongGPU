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

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <stdlib.h> //qsort

#include "config.hpp"
#include "coder.hpp"
#include "adaptor.hpp"
#include "dgraph.hpp"
#include "rcache.hpp"
#include "engine.hpp"
#include "query.hpp"

#include "mymath.hpp"
#include "timer.hpp"

using namespace std;


// a vector of pointers of all local engines
class GPU_Engine;
std::vector<GPU_Engine *> gpu_engines;


class GPU_Engine {
    std::vector<request_or_reply> msg_fast_path;
    Reply_Map rmap; // a map of replies for pending (fork-join) queries

    DGraph *graph;

    int sub_req_cnt;

    // all of these means const predicate
    void const_to_unknown(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        std::vector<int> updated_result_table;

        // the query plan is wrong
        assert ((req.get_col_num() == 0) && (req.get_col_num() == req.var2column(end)));

        int sz = 0;
        edge_t *res = graph->get_edges_global(tid, start, d, pid, &sz);
        for (int k = 0; k < sz; k++)
            updated_result_table.push_back(res[k].val);

        req.result_table.swap(updated_result_table);
        req.set_col_num(1);
        req.step++;
    }

    void const_to_known(request_or_reply &req) { assert(false); } /// TODO

    void known_to_unknown(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        std::vector<int> updated_result_table;

        // the query plan is wrong
        assert (req.get_col_num() == req.var2column(end));

        updated_result_table = cache->known_to_unknown(req,start,d,pid);

        req.set_col_num(req.get_col_num() + 1);
        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void known_to_known(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

        updated_result_table = cache->known_to_known(req,start,d,pid,end);

        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void known_to_const(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

        updated_result_table = cache->known_to_const(req,start,d,pid,end);

        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void index_to_unknown(request_or_reply &req) {
        int64_t idx = req.cmd_chains[req.step * 4];
        int64_t nothing = req.cmd_chains[req.step * 4 + 1];
        int64_t d = req.cmd_chains[req.step * 4 + 2];
        int64_t var = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

        // the query plan is wrong
        assert(req.get_col_num() == 0 && req.get_col_num() == req.var2column(var));

        updated_result_table = cache->index_to_unknown(req,idx,d);

        req.result_table.swap(updated_result_table);
        req.set_col_num(1);
        req.step++;
        req.local_var = -1;
    }

    vector<request_or_reply> generate_sub_query(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];

        vector<request_or_reply> sub_reqs;
        int num_sub_request = global_num_servers;
        sub_reqs.resize(num_sub_request);
        for (int i = 0; i < sub_reqs.size(); i++) {
            sub_reqs[i].comp_dev = GPU_comp;
            sub_reqs[i].pid = req.id;
            sub_reqs[i].cmd_chains = req.cmd_chains;
            sub_reqs[i].step = req.step;
            sub_reqs[i].col_num = req.col_num;
            sub_reqs[i].blind = req.blind;
            sub_reqs[i].local_var = start;
            sub_reqs[i].preds = req.preds;


            sub_reqs[i].sub_req = true;
        }

        //disable fast copy for single machine
        if (false && global_num_servers == 1) {
            //only single machine, no need to do any operation
            sub_reqs[0].gpu_history_table_ptr = req.gpu_history_table_ptr;
            sub_reqs[0].gpu_history_table_size = req.gpu_history_table_size;
        } else {
            //work seperation on cluster
            vector<int*> gpu_sub_table_ptr_list(num_sub_request);
            vector<int> gpu_sub_table_size_list(num_sub_request);
            cache->generate_sub_query(req,
                    start,
                    num_sub_request,
                    &gpu_sub_table_ptr_list[0],
                    &gpu_sub_table_size_list[0]);
            for (int i=0; i<num_sub_request; ++i) {
                sub_reqs[i].gpu_history_table_ptr = (char*)gpu_sub_table_ptr_list[i];
                //record orginial buffer head so buffer will not be shorten
                sub_reqs[i].gpu_origin_buffer_head = (char*)gpu_sub_table_ptr_list[0];
                sub_reqs[i].gpu_history_table_size = gpu_sub_table_size_list[i];

                // printf("GPU_Engine[%d]: sub_reqs[%d] history size %d\n", sid, i, sub_reqs[i].gpu_history_table_size);
            }
        }
        return sub_reqs;
    }

    // fork-join or in-place execution
    bool need_fork_join(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        return ((req.local_var != start)
                && (req.get_row_num() > 0));
    }

    bool execute_one_step(request_or_reply &req) {
        if (req.is_finished()) {
            return false;
        }
        if (req.step == 0 && req.start_from_index()) {
            index_to_unknown(req);
            return true;
        }
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t predicate = req.cmd_chains[req.step * 4 + 1];
        int64_t direction = req.cmd_chains[req.step * 4 + 2];
        int64_t end = req.cmd_chains[req.step * 4 + 3];

        if (predicate < 0) {
#ifdef VERSATILE
            switch (var_pair(req.variable_type(start), req.variable_type(end))) {
            case var_pair(const_var, unknown_var):
                const_unknown_unknown(req);
                break;
            case var_pair(known_var, unknown_var):
                known_unknown_unknown(req);
                break;
            default :
                assert(false);
                break;
            }
            return true;
#else
            cout << "ERROR: unsupport variable at predicate." << endl;
            cout << "Please add definition VERSATILE in CMakeLists.txt." << endl;
            assert(false);
#endif
        }

        // known_predicate
        switch (var_pair(req.variable_type(start), req.variable_type(end))) {
        // start from const_var
        case var_pair(const_var, const_var):
            cout << "ERROR: unsupported triple pattern (from const_var to const_var)" << endl;
            assert(false);
        case var_pair(const_var, unknown_var):
            const_to_unknown(req);
            break;
        case var_pair(const_var, known_var):
            cout << "ERROR: unsupported triple pattern (from const_var to known_var)" << endl;
            assert(false);

        // start from known_var
        case var_pair(known_var, const_var):
            known_to_const(req);
            break;
        case var_pair(known_var, known_var):
            known_to_known(req);
            break;
        case var_pair(known_var, unknown_var):
            known_to_unknown(req);
            break;

        // start from unknown_var
        case var_pair(unknown_var, const_var):
        case var_pair(unknown_var, known_var):
        case var_pair(unknown_var, unknown_var):
            cout << "ERROR: unsupported triple pattern (from unknown_var)" << endl;
            assert(false);

        default :
            assert(false);
        }

        return true;
    }


    void execute_request(request_or_reply &req) {
        uint64_t t1, t2;

        while (true) {
            t1 = timer::get_usec();
            execute_one_step(req);
            t2 = timer::get_usec();

            if (req.is_finished()) {
                req.row_num = req.get_row_num();


                if (req.blind)
                    req.clear_data(); // avoid take back the resuts

                adaptor->send(coder.sid_of(req.pid), coder.tid_of(req.pid), req);
#ifdef PIPELINE
                cache->shardmanager->reset();
#endif
                return;
            }

            if (need_fork_join(req)) {
                vector<request_or_reply> sub_rs = generate_sub_query(req);
                sub_req_cnt += 1;
                rmap.put_parent_request(req, sub_rs.size());
                for (int i = 0; i < sub_rs.size(); i++) {
                    if (i != sid) {
                        //adaptor->send(i, tid, sub_rs[i]);
                        adaptor->gpu_send(i, sub_rs[i]);
                        // increase count of GPU Engine task queue
                    } else {
                        msg_fast_path.push_back(sub_rs[i]);
                    }
                }
                return;
            }
        }
        return;
    }

    void execute(request_or_reply &r, GPU_Engine *gpu_engine) {
        if (r.is_request()) {
            // request
            r.id = coder.get_and_inc_qid();
            execute_request(r);
        } else {
            // reply
            gpu_engine->rmap.put_reply(r);
            if (gpu_engine->rmap.is_ready(r.pid)) {
                request_or_reply reply = gpu_engine->rmap.get_merged_reply(r.pid);
                adaptor->send(coder.sid_of(reply.pid), coder.tid_of(reply.pid), reply);
            }
        }
    }


public:
    int devid;
    int sid;    // server id
    int tid;    // thread id

    RCache *cache;
    Adaptor *adaptor;

    Coder coder;

    uint64_t last_time; // busy or not (work-oblige)


    GPU_Engine(int devid, int sid, int tid, RCache *cache, Adaptor *adaptor)
        : devid(devid), sid(sid), tid(tid), cache(cache), adaptor(adaptor),
          coder(sid, tid), last_time(0) {
        // CUDA_SAFE_CALL( cudaSetDevice(devid) );
        graph = cache->dgraph;
        sub_req_cnt = 0;
    }

    // Siyuan: GPU engine doesn't have work-stealing
    void run() {
        printf("GPU_Engine: sid=%d, tid=%d\n", sid, tid);
        while (true) {
            request_or_reply r;
            bool success;

            // fast path
            last_time = timer::get_usec();
            success = false;

            if (msg_fast_path.size() > 0) {
                r = msg_fast_path.back();
                msg_fast_path.pop_back();
                success = true;
            }

            if (success) {
                execute(r, gpu_engines[0]);
                continue; // fast-path priority
            }

            // normal path
            last_time = timer::get_usec();

            // own queue
            //success = adaptor->tryrecv(r);
            success = adaptor->gpu_tryrecv(r, (char*)cache->d_result_table, GPU_BUF_SIZE(sizeof(int)) );
            if (success && r.start_from_index()) {
                msg_fast_path.push_back(r);
                success = false;
            }

            if (success) execute(r, gpu_engines[0]);

        }
    }
};
