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

#include <vector>
#include <list>
#include <cassert>
#include "coder.hpp"
#include "reply_map.hpp"
#include "query.hpp"
#include "gpu.hpp"

#include "coder.hpp"
#include "agent_adaptor.hpp"
#include "dgraph.hpp"
#include "rcache.hpp"
#include "engine.hpp"
#include "query.hpp"

#include "mymath.hpp"
#include "timer.hpp"

// a vector of pointers of all local engines
class GPU_Engine;
std::vector<GPU_Engine *> gpu_engines;
uint64_t global_begin_time = 0, global_end_time = 0;

extern bool global_disable_gpudirect;

class GPU_Engine {

    std::vector<request_or_reply> msg_fast_path;
    Reply_Map rmap; // a map of replies for pending (fork-join) queries
    DGraph *graph;
    list<Pending_Msg> pending_msgs;
    bool used_pending_queue;
    char *host_bufs[11];

    int gen_next_worker_id() {
        static int last_tid = tid;
        int next_tid;

        next_tid = last_tid + 1;
        if (next_tid > TID_LAST_WORKER)
            next_tid = TID_FIRST_WORKER;

        last_tid = next_tid;
        return next_tid;
    }

    void sweep_msgs() {
        if (pending_msgs.empty()) return;

        bool success = false;
        for (auto it = pending_msgs.begin(); it != pending_msgs.end(); ) {
            assert(it->r.query_type == FULL_QUERY);

            success = adaptor->send(it->sid, it->tid, it->r);

            if (success) {
#ifdef QUERY_DEBUG
                printf("GPU_Engine[%d:%d] sweep_msgs(): sent to remote engine tid=%d success\n", sid, tid, it->tid);
#endif
                it = pending_msgs.erase(it);
            } else {

                if (global_sweep_send_to_engines) {
                    // if it is a reply, we cannot modify its destination tid
                    if (!it->r.is_request()) {
                        it++;
                        continue;
                    }

                    it->tid = gen_next_worker_id();
                    assert(it->tid >= global_num_proxies && it->tid < global_num_threads);

                    success = adaptor->send(it->sid, it->tid, it->r);
                    if (success) {
#ifdef QUERY_DEBUG
                        printf("GPU_Engine[%d:%d] sweep_msgs(): sent to remote engine tid=%d success\n", sid, tid, it->tid);
#endif
                        it = pending_msgs.erase(it);
                    } else {
                        it++;
                    }

                } else {
                    it++;
                }

            }
        }

    }

    void send_request(int dst_sid, int dst_tid, request_or_reply &r) {
        bool success;
        if (r.query_type == SPLIT_QUERY) {
            // assert(global_disable_gpudirect == false);
            assert(r.gpu_history_ptr != nullptr);
            assert(r.gpu_history_table_size > 0);
            if (r.gpu_history_ptr == (char *)0xdeadbeef) {
                success = adaptor->send_split(dst_sid, dst_tid, r, host_bufs[dst_sid], r.gpu_history_table_size);
            } else {
                success = adaptor->send_split(dst_sid, dst_tid, r, r.gpu_history_ptr, r.gpu_history_table_size);
            }
        } else {
            success = adaptor->send(dst_sid, dst_tid, r);
        }

        if (success) return;

        // copy history to host
        if (r.query_type == SPLIT_QUERY && r.gpu_history_ptr) {
            assert(r.gpu_history_table_size > 0);
            copy_history_to_host(r);
        }

        used_pending_queue = true;
        // save to pending queue
        pending_msgs.push_back(Pending_Msg(dst_sid, dst_tid, r));
    }



public:
    int devid;  // GPU device id
    int sid;    // server id
    int tid;    // thread id

    RCache *cache;
    Agent_Adaptor *adaptor;
    Coder coder;

    GPU_Engine(int devid, int sid, int tid, RCache *cache, Agent_Adaptor *adaptor)
        : devid(devid), sid(sid), tid(tid), cache(cache), adaptor(adaptor),
          coder(sid, tid) {
        graph = cache->dgraph;
        used_pending_queue = false;
        // host_bufs.resize(global_num_servers);
        // alloc host buffers for history of sub-queries
        int bufsz = global_rdma_buf_size_mb * 1024 * 1024;
        for (int i = 0; i < global_num_servers; ++i) {
            CUDA_SAFE_CALL( cudaMallocHost((void**)&host_bufs[i], bufsz, 0) );
            memset(host_bufs[i], 0, bufsz);
        }
    }

    bool get_used_pending_queue() {
        return used_pending_queue;
    }

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
#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_unknown(): [begin] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif

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

#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_unknown(): [end] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif
    }

    void known_to_known(request_or_reply &req) {
#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_known(): [begin] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

        updated_result_table = cache->known_to_known(req,start,d,pid,end);

        req.result_table.swap(updated_result_table);
        req.step++;

#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_known(): [end] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif
    }

    void known_to_const(request_or_reply &req) {
#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_const(): [begin] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif

        int64_t start = req.cmd_chains[req.step * 4];
        int64_t pid   = req.cmd_chains[req.step * 4 + 1];
        int64_t d     = req.cmd_chains[req.step * 4 + 2];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

        updated_result_table = cache->known_to_const(req,start,d,pid,end);

        req.result_table.swap(updated_result_table);
        req.step++;

#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:known_to_const(): [end] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif
    }

    void index_to_unknown(request_or_reply &req) {
        int64_t idx = req.cmd_chains[req.step * 4];
        int64_t nothing = req.cmd_chains[req.step * 4 + 1];
        int64_t d = req.cmd_chains[req.step * 4 + 2];
        int64_t var = req.cmd_chains[req.step * 4 + 3];
        vector<int> updated_result_table;

#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:index_to_unknown(): [begin] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif

        // the query plan is wrong
        assert(req.get_col_num() == 0 && req.get_col_num() == req.var2column(var));

        updated_result_table = cache->index_to_unknown(req,idx,d);

        req.result_table.swap(updated_result_table);
        req.set_col_num(1);
        req.step++;
        req.local_var = -1;

#ifdef QUERY_DEBUG
        printf("GPU_Engine[%d:%d]:index_to_unknown(): [end] reqid=%d, req_pid=%d, step=%d, table_size=%d\n",
                sid, tid, req.id, req.pid, req.step, req.get_table_size() );
#endif
    }

    vector<request_or_reply> generate_sub_query(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        int64_t end   = req.cmd_chains[req.step * 4 + 3];
        uint64_t t1, t2;

        int num_sub_request = global_num_servers;
        vector<request_or_reply> sub_reqs(num_sub_request);

        for (int i = 0; i < sub_reqs.size(); i++) {
            request_or_reply &r = sub_reqs[i];
            r.comp_dev = GPU_comp;
            r.pid = req.id;
            r.cmd_chains = req.cmd_chains;
            r.step = req.step;
            r.col_num = req.col_num;
            r.blind = req.blind;
            r.local_var = start;
            r.preds = req.preds;

            r.sub_req = true;
            if (i != sid) {
                r.query_type = SPLIT_QUERY;
            }
        }

        //disable fast copy for single machine
        if (num_sub_request == 1) {
            //only single machine, no need to do any operation
            sub_reqs[0].gpu_history_ptr = req.gpu_history_ptr;
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

            // t1 = timer::get_usec();
            for (int i=0; i<num_sub_request; ++i) {
                request_or_reply &r = sub_reqs[i];
                r.gpu_history_ptr = (char*)gpu_sub_table_ptr_list[i];
                //record orginial buffer head so buffer will not be shorten
                r.gpu_origin_buffer_head = (char*)gpu_sub_table_ptr_list[0];
                r.gpu_history_table_size = gpu_sub_table_size_list[i];
                // if gpu history table is empty, set it to FULL_QUERY, which
                // will be sent by normal RDMA
                if (i != sid && r.gpu_history_table_size == 0) {
                    r.query_type = FULL_QUERY;
                    r.gpu_history_ptr = nullptr;
                }
                else if (i != sid && global_disable_gpudirect) {
                    CUDA_SAFE_CALL(cudaMemcpy(host_bufs[i],
                                  r.gpu_history_ptr,
                                  sizeof(int) * r.gpu_history_table_size,
                                  cudaMemcpyDeviceToHost));
                    r.gpu_history_ptr = (char *)0xdeadbeef;
                    // 统计有多少数据会走GDR，这里统计的是单机的数据
                    global_stat.gdr_data_sz += (sizeof(int) * r.gpu_history_table_size);
                    global_stat.gdr_times += 1;
                }
            }

#ifdef WO_GDR_TEST
            // Siyuan: 测试w/o GPUDirect的代码
            t2 = timer::get_usec();
            if (sid == 0 && tid == global_num_threads - 1) {
                printf("GPU_Engine[%d:%d]: copied history to host. Ta time=%lu us\n",
                        sid, tid, t2 - t1);
            }
#endif
        }

        return sub_reqs;
    }

    void copy_history_to_host(request_or_reply &r) {
        if (r.gpu_history_table_size > 0) {
            CUDA_SAFE_CALL(cudaMemcpy(cache->h_result_table_buffer,
                                  r.gpu_history_ptr,
                                  sizeof(int) * r.gpu_history_table_size,
                                  cudaMemcpyDeviceToHost));
            r.result_table = vector<int>(cache->h_result_table_buffer, cache->h_result_table_buffer + r.gpu_history_table_size);
            assert(r.result_table.size() > 0);
            // Siyuan: 把history拷贝回host后，control和data就全都在一起了
            r.query_type = FULL_QUERY;
            r.gpu_history_ptr = nullptr;
            r.gpu_history_table_size = 0;
            r.gpu_origin_buffer_head = nullptr;
        }
    }

    // fork-join or in-place execution
    bool need_fork_join(request_or_reply &req) {
        int64_t start = req.cmd_chains[req.step * 4];
        return ((req.local_var != start)
                && (req.get_row_num() > 0));
    }

    bool execute_one_step(request_or_reply &req) {
        assert(req.comp_dev == GPU_comp);

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


    // 现在在cpu上通过GPU::query_id()就能查到当前在GPU上跑的query
    void execute_request(request_or_reply &req) {
        bool success;

        while (true) {
            // ensure history is on GPU
            assert(req.id == GPU::instance().query_id());

            execute_one_step(req);

            if (req.is_finished()) {
                req.row_num = req.get_row_num();
                req.query_type = FULL_QUERY;

                if (req.blind) {
                    req.clear_data(); // avoid take back the resuts
                }

                assert(req.is_request() == false);
                int psid, ptid;
                psid = coder.sid_of(req.pid);
                ptid = coder.tid_of(req.pid);
                send_request(psid, ptid, req);

#ifdef QUERY_DEBUG
                printf("GPU_Engine[%d:%d]: send reply#%d back to [sid=%d, tid=%d]\n", sid, tid, req.id, psid, ptid);
#endif

#ifdef PIPELINE
                cache->shardmanager->reset();
#endif
                return;
            }
            uint64_t t1,t2;

            if (need_fork_join(req)) {
                vector<request_or_reply> sub_rs = generate_sub_query(req);

                rmap.put_parent_request(req, sub_rs.size());
                // t1 = timer::get_usec();
                for (int i = 0; i < sub_rs.size(); i++) {
                    if (i != sid) {
                        // adaptor->send_split(i, tid, sub_rs[i]);
                        send_request(i, tid, sub_rs[i]);
                    } else {
                        msg_fast_path.push_back(sub_rs[i]);
                    }
                }
                // t2 = timer::get_usec();
                // if (sid == 0 && tid == global_num_threads - 1)
                    // printf("GPU_Engine[%d:%d] finish sending sub-queries. Tb time: %luus\n", sid, tid, t2 - t1);
                return;
            }
        }

        return;
    }

    void execute(request_or_reply &r, GPU_Engine *gpu_engine) {
        bool success;
        uint64_t t1,t2;
        if (r.is_request()) {
            // request
            r.id = coder.get_and_inc_qid();

            // #1 load history data to GPU
            if (r.query_type == SPLIT_QUERY && (r.gpu_history_ptr != nullptr && r.gpu_history_table_size > 0)) {
                GPU::instance().set_query_id(r.id);

            } else if (r.is_first_handler() || r.id != GPU::instance().query_id()) {
                if (!r.result_table.empty()) {
                    r.gpu_history_ptr = GPU::instance().load_history_data(r.host_table_ptr(), r.host_table_size());
                    // if (sid == 1 && tid == global_num_threads - 1) {
                        // printf("GPU_Engine[%d:%d]: copied history to GPU. time=%lu us\n",
                                // sid, tid, t2 - t1);
                    // }
                    assert(r.gpu_history_ptr != nullptr);
                    // dont forget to set GPU table size!
                    r.gpu_history_table_size = r.host_table_size();
                    r.result_table.clear();
                }
                GPU::instance().set_query_id(r.id);
            } else {
                printf("GPU_Engine[%d:%d]: WARNING: history data is already loaded!, reqid=%d, table_size=%d\n",
                        sid, tid, r.id, r.get_table_size());
            }

            execute_request(r);
            GPU::instance().clear_query_id(r.id);

        } else {
            // reply
            gpu_engine->rmap.put_reply(r);
            if (gpu_engine->rmap.is_ready(r.pid)) {
                request_or_reply reply = gpu_engine->rmap.get_merged_reply(r.pid);
                reply.query_type = FULL_QUERY;
                int psid = coder.sid_of(reply.pid);
                int ptid = coder.tid_of(reply.pid);
                assert(reply.is_request() == false);
#ifdef WO_GDR_TEST
                // Siyuan: 测试w/o GPUDirect的代码
                // 把merged reply发回给master proxy，表示最初发过来的heavy query我处理完了
                if (psid == 0 && ptid == 0) {
                    if (sid == 0 && tid == global_num_threads - 1) {
                        assert(global_end_time == 0);
                        global_end_time = timer::get_usec();
                        printf("GPU_Engine[%d:%d]: end timer!\n", sid, tid);
                        printf("GPU_Engine[%d:%d]: send reply#%d back to proxy[sid=%d, tid=%d]. latency=%luus\n",
                                sid, tid, reply.id, psid, ptid, global_end_time - global_begin_time);

                        global_begin_time = global_end_time = 0;
                    }
                }
#endif
                send_request(psid, ptid, reply);
            }
        }

    }


    // Siyuan: GPU engine doesn't have work-stealing
    void run() {
        printf("GPU_Engine[%d:%d] is running %s GPUDirect\n", sid, tid, (global_disable_gpudirect ? "w/o" : "w/"));
        uint64_t t1,t2;

        while (true) {
            request_or_reply r;
            bool success;

            // fast path
            success = false;

            if (msg_fast_path.size() > 0) {
                r = msg_fast_path.back();
                msg_fast_path.pop_back();
                success = true;
            }

            if (success) {
                execute(r, this);
                continue; // fast-path priority
            }

            // t1 = timer::get_usec();
            success = adaptor->tryrecv_split(r);
            // heavy query
            if (success && r.start_from_index()) {
                // t2 = timer::get_usec();
                // if (global_begin_time == 0) {
                    // global_begin_time = timer::get_usec();
                    // printf("GPU_Engine[%d:%d]: start timer!\n", sid, tid);
                // }
                // if ((r.is_request() && r.query_type == SPLIT_QUERY && r.gpu_history_ptr != nullptr)
                        // && (sid == 1 && tid == global_num_threads - 1)) {
                    // printf("GPU_Engine[%d:%d]: received_history_to_device. Tc time: %lu us\n",
                            // sid, tid, t2 - t1);
                // }
                msg_fast_path.push_back(r);
                success = false;
            }
            // check and send pending messages
            sweep_msgs();

            // light query
            if (success) execute(r, this);
        }
    }

};

