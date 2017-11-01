/*
 *
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
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpu_config.h"
#include "gpu_hash.h"
#include "gpu_stream.hpp"
#include "dgraph.hpp"
#include "rdf_meta.hpp"
#include "shard_manager.hpp"
#include "unit.hpp"

using namespace std;


//make color for profiling info
const string fg_red("\033[0;31m");
const string fg_green("\033[0;32m");
const string fg_yellow("\033[0;33m");
const string fg_blue("\033[0;34m");
const string fg_reset("\033[0m");
const string fg_null("");

#define mark_and_print_handler_start(handler_name) if(global_gpu_profiling) cout << fg_yellow << "Enter handler: " \
         << fg_blue << #handler_name \
         << fg_reset << endl; \
    uint64_t start_##handler_name=timer::get_usec();
#define mark_and_print_handler_end(handler_name) uint64_t end_##handler_name=timer::get_usec();\
    if(global_gpu_profiling) cout << fg_yellow << "Leave handler: " \
         << fg_blue << #handler_name \
         << fg_reset << endl; 
#define print_handler_interval(handler_name) if(global_gpu_profiling) cout << fg_yellow << "Total time in this handler: "\
         << fg_green <<end_##handler_name-start_##handler_name \
         << fg_yellow << " us" << fg_reset << endl;
#define set_timer_start(timer_x) uint64_t start_##timer_x=timer::get_usec();
#define set_timer_end(timer_x) uint64_t end_##timer_x=timer::get_usec();
#define cuda_sync() if(global_gpu_profiling) cudaDeviceSynchronize();
#define print_interval(name, indent) if(global_gpu_profiling) \
        cout << fg_yellow << indent << #name<<" time: " \
              << fg_red << end_##name-start_##name \
              << fg_yellow << " us" << fg_reset << endl;
 


class RCache {

public:

    //kv store on sysmem
    vertex_t* vertex_addr;
    edge_t* edge_addr;
    uint64_t num_gpu_slots = 0;
    uint64_t num_gpu_buckets = 0;
    uint64_t num_gpu_entries = 0;

    //kv store on gpumem
    vertex_t* d_vertex_addr;
    edge_t* d_edge_addr;

    //draft memory
    int* d_result_table; 
    int* d_updated_result_table; 
    ikey_t* d_key_list; 
    uint64_t* d_slot_id_list; 
    int* d_index_list; 
    int* d_index_list_mirror; // number of edges of key
    uint64_t* d_off_list;     // edges offset of key on GPU
    uint64_t* d_vertex_headers;
    uint64_t* d_edge_headers;

    int* h_result_table_buffer;

    pred_meta_t* d_pred_metas;

    StreamPool* streamPool;

    //streaming
    // vector<cudaStream_t> streams;
    cudaStream_t D2H_stream;
    vector<cudaEvent_t> events_compute;

    DGraph* dgraph;
    ShardManager* shardmanager;
    int sid;
    int devid;

    vector<pred_meta_t> pred_metas;

    RCache(int devid, DGraph* graph, StreamPool* streamPool, int sid)
		: devid(devid), dgraph(graph), streamPool(streamPool), sid(sid) {


        vertex_addr = dgraph->gstore.get_vertices_ptr();
        edge_addr = dgraph->gstore.get_edges_ptr();
        num_gpu_slots = global_gpu_num_keys_million * 1000 * 1000;
        num_gpu_buckets = num_gpu_slots/ASSOCIATIVITY;
        num_gpu_entries =  (GiB2B(global_gpu_memstore_size_gb) - num_gpu_slots * sizeof(vertex_t)) / sizeof(edge_t);
        pred_metas = dgraph->gstore.get_pred_metas();

        //migrate kv store from sysmem to gpumem
        const uint64_t vertex_array_size = sizeof(vertex_t)*dgraph->gstore.get_num_slots();
        const uint64_t edge_array_size = sizeof(edge_t)* dgraph->gstore.get_last_entry();
        const uint64_t gpu_vertex_array_size = sizeof(vertex_t)*num_gpu_slots;
        const uint64_t gpu_edge_array_size = sizeof(edge_t)*num_gpu_entries;
        cout << "num_gpu_slots:"<<num_gpu_slots<<endl;
        cout << "num_gpu_entries:"<<num_gpu_entries<<endl;
        cout << "GPU devid:" << devid << endl;

        CUDA_SAFE_CALL( cudaSetDevice(devid) );
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_vertex_addr, gpu_vertex_array_size ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_edge_addr, gpu_edge_array_size ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_result_table, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_updated_result_table, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_key_list, GPU_BUF_SIZE(sizeof(ikey_t)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_slot_id_list, GPU_BUF_SIZE(sizeof(uint64_t)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_index_list, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_index_list_mirror, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_off_list, GPU_BUF_SIZE(sizeof(uint64_t)) ));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_vertex_headers, sizeof(uint64_t)* NGPU_SHARDS));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_edge_headers, sizeof(uint64_t)* NGPU_SHARDS));

        CUDA_SAFE_CALL(cudaMemset( d_result_table, 0, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMemset( d_updated_result_table, 0, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMemset( d_key_list,0, GPU_BUF_SIZE(sizeof(ikey_t)) ));
        CUDA_SAFE_CALL(cudaMemset( d_slot_id_list, 0, GPU_BUF_SIZE(sizeof(uint64_t)) ));
        CUDA_SAFE_CALL(cudaMemset( d_index_list,0, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMemset( d_index_list_mirror,0, GPU_BUF_SIZE(sizeof(int)) ));
        CUDA_SAFE_CALL(cudaMemset( d_off_list, 0, GPU_BUF_SIZE(sizeof(uint64_t)) ));
        CUDA_SAFE_CALL(cudaMemset( d_vertex_headers, 0, sizeof(uint64_t) * NGPU_SHARDS));
        CUDA_SAFE_CALL(cudaMemset( d_edge_headers, 0, sizeof(uint64_t) * NGPU_SHARDS));

        //load meta data info
        int predicate_num = dgraph->gstore.get_num_preds() + 1;

#ifdef WUKONG_DEBUG
        printf("[INFO#%d] RCache: predicate_num=%d\n", sid, predicate_num);
#endif
        CUDA_SAFE_CALL(cudaMalloc( (void**)&d_pred_metas, sizeof(pred_meta_t)*predicate_num )); 
        CUDA_SAFE_CALL(cudaMemcpy( d_pred_metas,
                                   &(pred_metas[0]),
                                   sizeof(pred_meta_t)*predicate_num,
                                   cudaMemcpyHostToDevice ));



        CUDA_SAFE_CALL( cudaStreamCreate(&D2H_stream) );

        shardmanager = new ShardManager(d_vertex_addr,d_edge_addr,vertex_addr,edge_addr,num_gpu_buckets,num_gpu_entries,pred_metas);

        //prepare host table buffer
        CUDA_SAFE_CALL(cudaMallocHost( (void**)&h_result_table_buffer, GPU_BUF_SIZE(sizeof(int)) ));
        memset(h_result_table_buffer, 0, GPU_BUF_SIZE(sizeof(int)) );

        //pin host storage buffer
        CUDA_SAFE_CALL(cudaHostRegister(vertex_addr,vertex_array_size,0));
        CUDA_SAFE_CALL(cudaHostRegister(edge_addr,edge_array_size,0));

        //load all predicates
        vector<int> pred_list;
        if(global_gpu_preload_predicts=="all")
        {
            pred_list.resize(pred_metas.size()-1);
            iota(pred_list.begin(),pred_list.end(),1);
        }
        else if(global_gpu_preload_predicts!="none")
        {
            std::vector<int> vect;
            std::stringstream ss(global_gpu_preload_predicts);
            int i;

            while (ss >> i)
            {
                vect.push_back(i);

                if (ss.peek() == ',' || ss.peek() == ' ') 
                    ss.ignore();
            }
            for (i=0; i< vect.size(); i++)
                pred_list.push_back(vect.at(i));
        }
        request_or_reply dumb_req;
        for (auto &pred : pred_list) {
            shardmanager->load_predicate(pred, 0, dumb_req, streamPool->get_stream(), false);
        }

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    }
    ~RCache(){
        delete shardmanager;
    }

    void const_to_unknown(){
    };

    vector<int> known_to_unknown(request_or_reply &req,int start,int direction,int predict) {
        //cout <<  "predict:"<<predict<<endl;
        //load_predicate_wrapper(predict);

        uint64_t t1, t2, t3, t4;
        int* raw_result_table = &req.result_table[0];
        int query_size = req.get_row_num();
        cudaStream_t stream = streamPool->get_stream(predict);
        // printf("[INFO#%d] known_to_unknown: query_size=%d, step=%d\n", sid, query_size, req.step + 1);

        if (query_size==0) {
            //give buffer back to pool
            if (req.gpu_history_table_ptr!=nullptr)
                d_result_table = (int*)req.gpu_history_table_ptr;

            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
            return vector<int> ();
        }


        if(!req.is_first_handler()) {
            d_result_table = (int*)req.gpu_history_table_ptr;
        }
        else {
            CUDA_SAFE_CALL(cudaMemcpy(d_result_table,
                          raw_result_table,
                          sizeof(int) * req.result_table.size(),
                          cudaMemcpyHostToDevice));
        }
        assert(d_result_table!=d_updated_result_table); 
        assert(d_result_table!=nullptr); 

        // Siyuan: before processing the query, we should ensure the data
        // of required predicates is loaded.
        t1 = timer::get_usec();
        if(!shardmanager->check_pred_exist(predict))
            shardmanager->load_predicate(predict,predict,req, stream, false);

        t2 = timer::get_usec();



        //preload next
        if (global_gpu_enable_pipeline) {
            while(!req.preds.empty() && shardmanager->check_pred_exist(req.preds[0])) {
                req.preds.erase(req.preds.begin());
            }
            if(!req.preds.empty()) {
                shardmanager->load_predicate(req.preds[0], predict, req, streamPool->get_stream(req.preds[0]), true);
            }
        }
        vector<uint64_t> vertex_headers = shardmanager->get_pred_vertex_headers(predict);
        vector<uint64_t> edge_headers = shardmanager->get_pred_edge_headers(predict);
        uint64_t pred_vertex_shard_size = shardmanager->shard_bucket_num;
        uint64_t pred_edge_shard_size = shardmanager->shard_entry_num;

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertex_headers,
                          &(vertex_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_vertex_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_edge_headers,
                          &(edge_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_edge_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));


#ifdef BREAK_DOWN
        t3 = timer::get_usec();
#endif

        // Siyuan: 这里有用到d_result_table里的数据
        generate_key_list_k2u(d_result_table,
                              start,
                              direction,
                              predict,
                              req.get_col_num(),
                              d_key_list,
                              query_size,
                              stream);
        t4 = timer::get_usec();
#ifdef BREAK_DOWN
        printf("[INFO#%d] known_to_unknown: generate_key_list_k2u() %luus\n", sid, t4 - t3);

        t3 = timer::get_usec();
#endif
        get_slot_id_list(d_vertex_addr,
                      d_key_list,
                      d_slot_id_list,
                      d_pred_metas,
                      d_vertex_headers,
                      pred_vertex_shard_size,
                      query_size,
                      stream);
        t4 = timer::get_usec();

#ifdef BREAK_DOWN
        printf("[INFO#%d] known_to_unknown: get_slot_id_list() %luus\n", sid, t4 - t3);

        t3 = timer::get_usec();
#endif
        get_edge_list(d_slot_id_list,
                      d_vertex_addr,
                      d_index_list,
                      d_index_list_mirror,
                      d_off_list,
                      pred_metas[predict].edge_start,
                      d_edge_headers,
                      pred_edge_shard_size,
                      query_size,
                      stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();
        printf("[INFO#%d] known_to_unknown: get_edge_list() %luus\n", sid, t4 - t3);


        t3 = timer::get_usec();
#endif
        calc_prefix_sum(d_index_list,
                        d_index_list_mirror,
                        query_size,
                        stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();
        printf("[INFO#%d] known_to_unknown: calc_prefix_sum() %luus\n", sid, t4 - t3);


        t3 = timer::get_usec();
#endif
        int table_size = update_result_table_k2u(d_result_table,
                                                           d_updated_result_table,
                                                           d_index_list,
                                                           d_off_list,
                                                           req.get_col_num(),
                                                           d_edge_addr,
                                                           d_edge_headers,
                                                           pred_edge_shard_size,
                                                           query_size,
                                                           stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();
        printf("[INFO#%d] known_to_unknown: update_result_table_k2u() %luus\n", sid, t4 - t3);
#endif


        //sync to get result
        CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

        t4 = timer::get_usec();

#ifdef PIPELINE
        printf(">>>> [known_to_unknown] step=%d, pid=%d, load time=%lluus, compute time=%lluus\n", req.step + 1, predict, t2-t1, t4-t3);
#endif



        vector<int> updated_result_table;
        // printf("[INFO#%d] known_to_unknown: table_size=%d, row=%d, col=%d step=%d\n", sid, table_size, req.get_row_num(), req.get_col_num(), req.step + 1);
        if (!req.is_last_handler()) {
            req.gpu_history_table_ptr = (char*)d_updated_result_table;
            req.gpu_history_table_size = table_size;

            if (req.gpu_origin_buffer_head != nullptr)
            {
                d_updated_result_table = (int*)req.gpu_origin_buffer_head;
                req.gpu_origin_buffer_head = nullptr;
            }
            else
                d_updated_result_table = d_result_table;

        } else {
            CUDA_SAFE_CALL(cudaMemcpy(h_result_table_buffer,
                                  d_updated_result_table,
                                  sizeof(int) * table_size,
                                  cudaMemcpyDeviceToHost));
            updated_result_table = vector<int>(h_result_table_buffer, h_result_table_buffer + table_size);

            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
        }
        return updated_result_table;
    };

    vector<int> known_to_known(request_or_reply &req,int start,int direction,int predict, int end){
        //cout <<  "predict:"<<predict<<endl;
        //load_predicate_wrapper(predict);

        uint64_t t1, t2, t3, t4;
        int* raw_result_table = &req.result_table[0];
        int query_size = req.get_row_num();
        cudaStream_t stream = streamPool->get_stream(predict);
        // cout <<"query_size:"<<query_size<<endl;

        if(query_size==0)
        {
            //give buffer back to pool
            if (req.gpu_history_table_ptr!=nullptr)
                d_result_table = (int*)req.gpu_history_table_ptr;

            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
            return vector<int> ();
        }


        if(!req.is_first_handler())
        {
            d_result_table = (int*)req.gpu_history_table_ptr;
        }
        else
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_result_table,
                          raw_result_table,
                          sizeof(int) * req.result_table.size(),
                          cudaMemcpyHostToDevice));
        }

        assert(d_result_table!=d_updated_result_table); 
        assert(d_result_table!=nullptr); 

        t1 = timer::get_usec();
        if(!shardmanager->check_pred_exist(predict))
            shardmanager->load_predicate(predict,predict,req,stream, false);

        t2 = timer::get_usec();

        //preload next, pipeline
        if (global_gpu_enable_pipeline)
        {
            while(!req.preds.empty() && shardmanager->check_pred_exist(req.preds[0]))
            {
                req.preds.erase(req.preds.begin());
            }
            if(!req.preds.empty())
                shardmanager->load_predicate(req.preds[0],predict,req, streamPool->get_stream(req.preds[0]), true);
        }
        vector<uint64_t> vertex_headers = shardmanager->get_pred_vertex_headers(predict);
        vector<uint64_t> edge_headers = shardmanager->get_pred_edge_headers(predict);
        uint64_t pred_vertex_shard_size = shardmanager->shard_bucket_num;
        uint64_t pred_edge_shard_size = shardmanager->shard_entry_num;

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertex_headers,
                          &(vertex_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_vertex_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_edge_headers,
                          &(edge_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_edge_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));

        t3 = timer::get_usec();

        generate_key_list_k2u(d_result_table,
                              start,
                              direction,
                              predict,
                              req.get_col_num(),
                              d_key_list,
                              query_size,
                              stream) ;

        get_slot_id_list(d_vertex_addr,
                      d_key_list,
                      d_slot_id_list,
                      d_pred_metas,
                      d_vertex_headers,
                      pred_vertex_shard_size,
                      query_size,
                      stream);

        get_edge_list_k2k(d_slot_id_list,
                      d_vertex_addr,
                      d_index_list,
                      d_index_list_mirror,
                      d_off_list,
                      query_size,
                      d_edge_addr,
                      d_result_table,
                      req.get_col_num(),
                      end,
                      pred_metas[predict].edge_start,
                      d_edge_headers,
                      pred_edge_shard_size,
                      stream);

        calc_prefix_sum(d_index_list,
                        d_index_list_mirror,
                        query_size,
                        stream);

        int table_size = update_result_table_k2k(d_result_table,
                                                           d_updated_result_table,
                                                           d_index_list,
                                                           d_off_list,
                                                           req.get_col_num(),
                                                           d_edge_addr,
                                                           end,
                                                           query_size,
                                                           stream);

        //sync to get result
        CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

        t4 = timer::get_usec();

#ifdef PIPELINE
        printf(">>>> [known_to_known] step=%d, pid=%d, load time=%lluus, compute time=%lluus\n", req.step + 1, predict, t2-t1, t4-t3);
#endif

        vector<int> updated_result_table;
        if(!req.is_last_handler())
        {
            req.gpu_history_table_ptr = (char*)d_updated_result_table;
            req.gpu_history_table_size = table_size;

            if (req.gpu_origin_buffer_head != nullptr)
            {
                d_updated_result_table = (int*)req.gpu_origin_buffer_head;
                req.gpu_origin_buffer_head = nullptr;
            }
            else
                d_updated_result_table = d_result_table;
        }
        else
        {
            CUDA_SAFE_CALL(cudaMemcpy(h_result_table_buffer,
                                  d_updated_result_table,
                                  sizeof(int) * table_size,
                                  cudaMemcpyDeviceToHost));
            updated_result_table = vector<int>(h_result_table_buffer, h_result_table_buffer + table_size);
            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
        }


        return updated_result_table;
    };

     vector<int> known_to_const(request_or_reply &req,int start,int direction,int predict, int end) {
        //cout <<  "predict:"<<predict<<endl;
        //load_predicate_wrapper(predict);
        cudaStream_t stream = streamPool->get_stream(predict);

        uint64_t t1, t2, t3, t4;
        int* raw_result_table = &req.result_table[0];
        int query_size = req.get_row_num();
        // cout <<"query_size:"<<query_size<<endl;

        if(query_size==0)
        {
            //give buffer back to pool
            if (req.gpu_history_table_ptr!=nullptr)
                d_result_table = (int*)req.gpu_history_table_ptr;

            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
            return vector<int> ();
        }


        if(!req.is_first_handler())
        {
            d_result_table = (int*)req.gpu_history_table_ptr;
        }
        else
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_result_table,
                          raw_result_table,
                          sizeof(int) * req.result_table.size(),
                          cudaMemcpyHostToDevice));
        }

        assert(d_result_table!=d_updated_result_table); 
        assert(d_result_table!=nullptr); 

        t1 = timer::get_usec();
        if(!shardmanager->check_pred_exist(predict))
            shardmanager->load_predicate(predict, predict, req, stream, false);


        t2 = timer::get_usec();

        // preload next predicate
        if(global_gpu_enable_pipeline)
        {
            while(!req.preds.empty() && shardmanager->check_pred_exist(req.preds[0]))
            {
                req.preds.erase(req.preds.begin());
            }
            if(!req.preds.empty())
                shardmanager->load_predicate(req.preds[0], predict, req, streamPool->get_stream(req.preds[0]), true);

        }
        vector<uint64_t> vertex_headers = shardmanager->get_pred_vertex_headers(predict);
        vector<uint64_t> edge_headers = shardmanager->get_pred_edge_headers(predict);
        uint64_t pred_vertex_shard_size = shardmanager->shard_bucket_num;
        uint64_t pred_edge_shard_size = shardmanager->shard_entry_num;

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertex_headers,
                          &(vertex_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_vertex_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));

        CUDA_SAFE_CALL(cudaMemcpyAsync(d_edge_headers,
                          &(edge_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_edge_shard_size[predict],
                          cudaMemcpyHostToDevice,
                          stream));

        t3 = timer::get_usec();

        generate_key_list_k2u(d_result_table,
                              start,
                              direction,
                              predict,
                              req.get_col_num(),
                              d_key_list,
                              query_size,
                              stream);

        get_slot_id_list(d_vertex_addr,
                      d_key_list,
                      d_slot_id_list,
                      d_pred_metas,
                      d_vertex_headers,
                      pred_vertex_shard_size,
                      query_size,
                      stream);

        get_edge_list_k2c(d_slot_id_list,
                      d_vertex_addr,
                      d_index_list,
                      d_index_list_mirror,
                      d_off_list,
                      query_size,
                      d_edge_addr,
                      end,
                      pred_metas[predict].edge_start,
                      d_edge_headers,
                      pred_edge_shard_size,
                      stream);

        calc_prefix_sum(d_index_list,
                        d_index_list_mirror,
                        query_size,
                        stream);

        int table_size = update_result_table_k2k(d_result_table,
                                                           d_updated_result_table,
                                                           d_index_list,
                                                           d_off_list,
                                                           req.get_col_num(),
                                                           d_edge_addr,
                                                           end,
                                                           query_size,
                                                           stream);
        //sync to get result
        CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

        t4 = timer::get_usec();

#ifdef PIPELINE
        printf(">>>> [known_to_const] step=%d, pid=%d, load time=%lluus, compute time=%lluus\n", req.step + 1, predict, t2-t1, t4-t3);
#endif

        vector<int> updated_result_table;
        if(!req.is_last_handler())
        {
            req.gpu_history_table_ptr = (char*)d_updated_result_table;
            req.gpu_history_table_size = table_size;

            if (req.gpu_origin_buffer_head != nullptr)
            {
                d_updated_result_table = (int*)req.gpu_origin_buffer_head;
                req.gpu_origin_buffer_head = nullptr;
            }
            else
                d_updated_result_table = d_result_table;
        }
        else
        {
            CUDA_SAFE_CALL(cudaMemcpy(h_result_table_buffer,
                                  d_updated_result_table,
                                  sizeof(int) * table_size,
                                  cudaMemcpyDeviceToHost));
            updated_result_table = vector<int>(h_result_table_buffer, h_result_table_buffer + table_size);
            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
        }

        return updated_result_table;
    };

    vector<int> index_to_unknown(request_or_reply &req, int index_vertex, int direction) {
        //cout << "index_vertex:"<<index_vertex<<endl;
        //load_predicate_wrapper(index_vertex);
        uint64_t t1, t2, t3, t4;
        int query_size = 1;
        cudaStream_t stream = streamPool->get_stream(index_vertex);

        // printf("[INFO#%d] index_to_unknown: query_size=%d, step=%d\n", sid, query_size, req.step);

        if (!req.is_first_handler()) {
            d_result_table = (int*)req.gpu_history_table_ptr;
        }

        assert(d_result_table!=d_updated_result_table);
        assert(d_result_table!=nullptr);

        t1 = timer::get_usec();

        if(!shardmanager->check_pred_exist(index_vertex))
            shardmanager->load_predicate(index_vertex,index_vertex,req,stream, false);

        t2 = timer::get_usec();

        //preload next
        if(global_gpu_enable_pipeline)
        {
            //for(auto pred:req.preds)
            //    if(!shardmanager->check_pred_exist(pred))
            //        shardmanager->load_predicate(req.preds[pred],req.preds,streams[req.preds[pred]]);
            while(!req.preds.empty() && shardmanager->check_pred_exist(req.preds[0]))
            {
                req.preds.erase(req.preds.begin());
            }
            if(!req.preds.empty())
                shardmanager->load_predicate(req.preds[0],index_vertex,req, streamPool->get_stream(req.preds[0]), true);
        }
        vector<uint64_t> vertex_headers = shardmanager->get_pred_vertex_headers(index_vertex);
        vector<uint64_t> edge_headers = shardmanager->get_pred_edge_headers(index_vertex);
        uint64_t pred_vertex_shard_size = shardmanager->shard_bucket_num;
        uint64_t pred_edge_shard_size = shardmanager->shard_entry_num;
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertex_headers,
                          &(vertex_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_vertex_shard_size[index_vertex],
                          cudaMemcpyHostToDevice,
                          stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_edge_headers,
                          &(edge_headers[0]),
                          sizeof(uint64_t) * shardmanager->pred_edge_shard_size[index_vertex],
                          cudaMemcpyHostToDevice,
                          stream));

        t3 = timer::get_usec();

        // Siyuan: 这个d_result_table并没有作为输入
        generate_key_list_i2u(d_result_table,
                              index_vertex,
                              direction,
                              d_key_list,
                              query_size,
                              stream) ;
#ifdef BREAK_DOWN
        t4 = timer::get_usec();
        printf("[INFO#%d] index_to_unknown: generate_key_list_i2u() %luus\n", sid, t4 - t3);


        t3 = timer::get_usec();
#endif
        get_slot_id_list(d_vertex_addr,
                      d_key_list,
                      d_slot_id_list,
                      d_pred_metas,
                      d_vertex_headers,
                      pred_vertex_shard_size,
                      query_size,
                      stream);
        t4 = timer::get_usec();

#ifdef BREAK_DOWN
        printf("[INFO#%d] index_to_unknown: get_slot_id_list() %luus\n", sid, t4 - t3);

        t3 = timer::get_usec();
#endif
        get_edge_list(d_slot_id_list,
                      d_vertex_addr,
                      d_index_list,
                      d_index_list_mirror,
                      d_off_list,
                      pred_metas[index_vertex].edge_start,
                      d_edge_headers,
                      pred_edge_shard_size,
                      query_size,
                      stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();

        printf("[INFO#%d] index_to_unknown: get_edge_list() %luus\n", sid, t4 - t3);

        t3 = timer::get_usec();
#endif
        calc_prefix_sum(d_index_list,
                        d_index_list_mirror,
                        query_size,
                        stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();

        printf("[INFO#%d] index_to_unknown: calc_prefix_sum() %luus\n", sid, t4 - t3);

        t3 = timer::get_usec();
#endif
        int table_size = update_result_table_i2u(d_result_table,
                                                           d_updated_result_table,
                                                           d_index_list,
                                                           d_off_list,
                                                           d_edge_addr,
                                                           d_edge_headers,
                                                           pred_edge_shard_size,
                                                           stream);
#ifdef BREAK_DOWN
        t4 = timer::get_usec();

        printf("[INFO#%d] index_to_unknown: update_result_table_i2u() %luus\n", sid, t4 - t3);
#endif
        //int *x = new int [table_size];
        //CUDA_SAFE_CALL(cudaMemcpy(x,
        //            d_updated_result_table,
        //            sizeof(int) * table_size,
        //            cudaMemcpyDeviceToHost));
        //for (int i =0;i<table_size;++i)
        //    cout <<x[i]<<endl;


        //sync to get result
        CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

        vector<int> updated_result_table;
        // Siyuan: 如果不是最后一个handler
        if(!req.is_last_handler())
        {
            req.gpu_history_table_ptr = (char*)d_updated_result_table;
            req.gpu_history_table_size = table_size;
            d_updated_result_table = d_result_table;
        }
        else
        {   // 若是最后一个handler
            CUDA_SAFE_CALL(cudaMemcpy(h_result_table_buffer,
                                  d_updated_result_table,
                                  sizeof(int) * table_size,
                                  cudaMemcpyDeviceToHost));
            updated_result_table = vector<int>(h_result_table_buffer, h_result_table_buffer + table_size);
            req.gpu_history_table_ptr = nullptr;
            req.gpu_history_table_size = 0;
        }
        return updated_result_table;
    };

    void generate_sub_query(request_or_reply &req,
                            int start,
                            int num_sub_request,
                            int** gpu_sub_table_ptr_list,
                            int* gpu_sub_table_size_list ) {
        // int* raw_result_table = &req.result_table[0];
        int query_size = req.get_row_num();

        // Siyuan: step肯定大于0
        assert(req.step > 0);

        // Siyuan: 如果不是第一个handler
        if (!req.is_first_handler()) {
            d_result_table = (int*)req.gpu_history_table_ptr;
        } else {
            assert(false);
        }

        assert(d_result_table!=d_updated_result_table);
        assert(d_result_table!=nullptr);

        // borrow buffers to store related data
        // Siyuan: d_mapping_list是position_list
        int* d_position_list = (int*)d_slot_id_list;
        int* d_server_id_list = d_index_list;
        int *d_server_sum_list=d_index_list_mirror;
        CUDA_SAFE_CALL( cudaMemsetAsync(d_server_sum_list, 0, num_sub_request * sizeof(int), D2H_stream) );

        calc_dispatched_position(d_result_table,
                                 d_position_list,
                                 d_server_id_list,
                                 d_server_sum_list,
                                 gpu_sub_table_size_list,
                                 start,
                                 req.get_col_num(),
                                 num_sub_request,
                                 query_size,
                                 D2H_stream);
        CUDA_SAFE_CALL( cudaStreamSynchronize(D2H_stream) );

        vector<int> gpu_sub_table_head_list(num_sub_request);


        CUDA_SAFE_CALL(cudaMemcpyAsync(&gpu_sub_table_head_list[0],
                                  d_server_sum_list,
                                  sizeof(int) * num_sub_request,
                                  cudaMemcpyDeviceToHost,
                                  D2H_stream));

        CUDA_SAFE_CALL( cudaStreamSynchronize(D2H_stream) );

        update_result_table_sub(d_result_table,
                                d_updated_result_table,
                                d_position_list,
                                d_server_id_list,
                                d_server_sum_list,
                                req.get_col_num(),
                                num_sub_request,
                                query_size,
                                D2H_stream);

        for (int i = 0; i < num_sub_request; ++i) {
            gpu_sub_table_size_list[i] *= req.get_col_num();
            gpu_sub_table_head_list[i] *= req.get_col_num();
            gpu_sub_table_ptr_list[i] = d_updated_result_table + gpu_sub_table_head_list[i];
        }

        //sync to get result
        CUDA_SAFE_CALL( cudaStreamSynchronize(D2H_stream) );

        d_updated_result_table = d_result_table;
        // Siyuan: Q: 为什么这2个buffer要换着用？
        // A: 因为这个handler的输出作为下一个handler的输入，所以这2个buffer轮流作为CUDA kernel
        // 的输入和输出buffer。
        /* if (!req.is_last_handler()) {
         *     d_updated_result_table = d_result_table;
         * } else {
         *     assert(false);
         * } */
    };

};
