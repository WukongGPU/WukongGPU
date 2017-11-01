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
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gpu_config.h"
#include "rdf_meta.hpp"
#include "gpu_hash.h"


using namespace std;

#define SHARD_ID_ERR -1

/**
 * A manager to allocate free slots to predicates
 *
 */
class ShardManager {
public:
    const int shard_num = 100;  // 总共100个shard

    int pred_num = 0;
    vector<pred_meta_t> pred_metas;

    //bitmap to record a shard is used
    vector<bool> vertex_bitmap;
    vector<bool> edge_bitmap;

    //
    vector<vector<int>> vertex_allocation;  // two-dim bitmap, 标记分配给每个predicate的shard是否是used
    vector<int> pred_vertex_shard_size;
    vector<vector<int>> edge_allocation;
    vector<int> pred_edge_shard_size;
    vertex_t* d_vertex_addr;
    edge_t* d_edge_addr;
    vertex_t* vertex_addr;
    edge_t* edge_addr;

    uint64_t total_bucket_num = 0;
    uint64_t total_entry_num = 0;
    uint64_t shard_bucket_num = 0;
    uint64_t shard_entry_num = 0;

    ShardManager(vertex_t* d_v_a, edge_t* d_e_a, vertex_t* v_a, edge_t* e_a, uint64_t num_buckets, uint64_t num_entries, vector<pred_meta_t> p_metas)
    {
        d_vertex_addr = d_v_a;
        d_edge_addr = d_e_a;
        vertex_addr = v_a;
        edge_addr = e_a;

        total_bucket_num = num_buckets;
        total_entry_num = num_entries;
        shard_bucket_num = total_bucket_num/shard_num;  // 每个bucket shard有多少个bucket
        shard_entry_num = total_entry_num/shard_num;    // 每个entry shard有多少个entry

        pred_metas = p_metas;
        pred_num = pred_metas.size();
        printf("[INFO] ShardManager: pred_num=%d, vertex_shard_sz=%dMB, entry_shard_sz=%dMB\n", pred_num,
                shard_bucket_num * ASSOCIATIVITY * sizeof(vertex_t) / (1024*1024),
                shard_entry_num * sizeof(edge_t) / (1024*1024) );

        //init bitmap
        int init_array[shard_num] = { 0 };
        vertex_bitmap.assign(&init_array[0],&init_array[0]+shard_num);
        edge_bitmap.assign(&init_array[0],&init_array[0]+shard_num);

        // for each predicate
        for (int i=0;i<pred_num;++i) {
            pred_meta_t pred = pred_metas[i];
            // Siyuan: vertex_len是main hdr和indirect hdr共需要的shard数量
            int vertex_len = ceil((double)(pred.partition_sz+pred.indrct_hdr_end-pred.indrct_hdr_start)/shard_bucket_num);
            vertex_allocation.push_back(vector<int>());
            // Siyuan: 把predicate的所有slot初始化为-1（表示没有分配shards）
            for(int j=0;j<vertex_len;++j)
                vertex_allocation[i].push_back(SHARD_ID_ERR);

            // 记录每个pid的vertex数据所需的shard数量
            pred_vertex_shard_size.push_back(vertex_len);

            int edge_len = ceil((double)(pred.edge_end-pred.edge_start)/shard_entry_num);
            edge_allocation.push_back(vector<int>());
            for(int j=0;j<edge_len;++j)
                edge_allocation[i].push_back(SHARD_ID_ERR);
            pred_edge_shard_size.push_back(edge_len);

        }
    }

    int get_next_free_vertex_shard(vector<int> conflicts, int pred_to_load, int pred_to_execute)
    {
        // #1 Siyuan: 先从free的shards中分配
        for (int i=0;i<shard_num;++i) {
            if (!vertex_bitmap[i])
            {
                vertex_bitmap[i] = true;
                return i;
            }
        }

        // #2 Siyuan: 没有free shards了，那么evict已经allocated的shards
        // 这个循环的范围是0~31，虽然pid=0是无效的predicate，但这也是无害的
        // 之前观察到vertex_allocation[18][0]是有分配block的，那为什么没有evict掉pid=18的shards呢？
        for (int i=0; i < pred_num; ++i) {
            // 在conflicts集合中找到了，那么skip这个predicate
            if (find(conflicts.begin(), conflicts.end(), i)!=conflicts.end())
                continue;

            if (i==pred_to_load) continue;
            if (i==pred_to_execute) continue;

            // Siyuan: pred_vertex_shard_size记录的是这个predicate占用了多少个shards
            for(int j=0;j<pred_vertex_shard_size[i];++j)
                if (vertex_allocation[i][j]!=SHARD_ID_ERR)
                {
                    int block_id = vertex_allocation[i][j];
                    vertex_allocation[i][j] = SHARD_ID_ERR;
#ifdef WUKONG_DEBUG
                    cout <<"get_free_vertex_shard(): evict: pid="<<i<<", block="<<j<<endl;
#endif
                    return block_id;
                }
        }


        //worst case: evict predicates related to query
        reverse(conflicts.begin(),conflicts.end());
        for(int i:conflicts) {
            if (i==pred_to_load) continue;
            if (i==pred_to_execute) continue;

            for(int j=0;j<pred_vertex_shard_size[i];++j)
                if (vertex_allocation[i][j]!=-1)
                {
                    int block_id = vertex_allocation[i][j];
                    vertex_allocation[i][j] = -1;
#ifdef WUKONG_DEBUG
                    cout <<"get_free_vertex_shard(): evict: pid="<<i<<", block="<<j<<endl;
#endif
                    return block_id;
                }
        }

        return -1;
    }

    int get_next_free_edge_shard(vector<int> conflicts, int pred_to_load, int pred_to_execute)
    {
        for(int i=0;i<shard_num;++i)
            if (!edge_bitmap[i])
            {
                edge_bitmap[i] = true;
                return i;
            }
                //evict
        for(int i=0;i<pred_num;++i)
        {
            if (find(conflicts.begin(), conflicts.end(),i)!=conflicts.end())
                continue;

            if (i==pred_to_load) continue;
            if (i==pred_to_execute) continue;

            for(int j=0;j<pred_edge_shard_size[i];++j)
                if (edge_allocation[i][j]!=-1)
                {
                    int block_id = edge_allocation[i][j];
                    edge_allocation[i][j] = -1;
#ifdef WUKONG_DEBUG
                    cout <<"get_free_edge_shard(): evict: pid="<<i<<", block="<<j<<endl;
#endif
                    return block_id;
                }
        }

        //worst case: evict predicates related to query
        reverse(conflicts.begin(),conflicts.end());
        for(int i:conflicts)
        {
            if (i==pred_to_load) continue;
            if (i==pred_to_execute) continue;

            for(int j=0;j<pred_edge_shard_size[i];++j)
                if (edge_allocation[i][j]!=-1)
                {
                    int block_id = edge_allocation[i][j];
                    edge_allocation[i][j] = -1;
#ifdef WUKONG_DEBUG
                    cout <<"get_free_edge_shard(): [worst case] evict: pid="<<i<<", block="<<j<<endl;
#endif
                    return block_id;
                }
        }

        return -1;
    }

    // Before processing the query, we should ensure the data of required predicates is loaded.
    // TODO: 这样一个一个slot的去看是否分配了shard太慢了，
    // 可以用一个带计数器的结构来替换掉vertex_allocation目前所用的二维vector。
    bool check_pred_exist(int pred)
    {
        for (auto vert:vertex_allocation[pred])
           if(vert==SHARD_ID_ERR)
               return false;

        for (auto edge:edge_allocation[pred])
           if(edge==SHARD_ID_ERR)
               return false;

        return true;
    }

    vector<uint64_t> get_pred_vertex_headers(int pred)
    {
        vector<uint64_t> headers;
        for(auto shard_id:vertex_allocation[pred])
            headers.push_back(shard_id*shard_bucket_num);
        return headers;
    }

    vector<uint64_t> get_pred_edge_headers(int pred)
    {
        vector<uint64_t> headers;
        for(auto shard_id:edge_allocation[pred])
            headers.push_back(shard_id*shard_entry_num);
        return headers;
    }

    // Siyuan: pred_slice_id 这是predicate的逻辑shard id，类似于inode的逻辑块号
    // shard_id是物理块号
    void load_vertex_shard(int pred, int pred_slice_id, int shard_id, cudaStream_t stream_id)
    {
        int end_main_slice_id = ceil((double)( pred_metas[pred].partition_sz)/shard_bucket_num)-1;
        int end_indrct_slice_id = ceil((double)(pred_metas[pred].partition_sz+pred_metas[pred].indrct_hdr_end-pred_metas[pred].indrct_hdr_start)/shard_bucket_num)-1;
        uint64_t tail_main_size = 0;
        uint64_t tail_indrct_size = 0;
        uint64_t tail_indrct_start = 0;
        uint64_t tail_indrct_head = 0;

        if (pred_slice_id==end_main_slice_id)
        {
            tail_main_size = (pred_metas[pred].partition_sz)%shard_bucket_num;
            tail_indrct_start = tail_main_size;
            tail_indrct_head = shard_bucket_num;
        }
        else if (pred_slice_id<end_main_slice_id)
            tail_main_size = shard_bucket_num;
        else
        {
            tail_indrct_head = shard_bucket_num - (pred_metas[pred].partition_sz)%shard_bucket_num;
            tail_main_size = 0;
        }

        // shard_id*shard_bucket_num定位到GPU上的起始地址
        if (tail_main_size!=0)
        CUDA_SAFE_CALL(cudaMemcpyAsync( d_vertex_addr+shard_id*shard_bucket_num*ASSOCIATIVITY,
                                   vertex_addr+(pred_metas[pred].main_hdr_start+pred_slice_id*shard_bucket_num)*ASSOCIATIVITY,
                                   sizeof(vertex_t)*tail_main_size*ASSOCIATIVITY,
                                   cudaMemcpyHostToDevice,
                                   stream_id ));

        if (pred_slice_id<end_main_slice_id)
            tail_indrct_size = 0;
        else if (pred_slice_id==end_indrct_slice_id)
        {
            tail_indrct_size = (pred_metas[pred].partition_sz+pred_metas[pred].indrct_hdr_end-pred_metas[pred].indrct_hdr_start)%shard_bucket_num-tail_indrct_start;
        }
        else if (pred_slice_id<end_indrct_slice_id)
        {
            tail_indrct_size = shard_bucket_num-tail_indrct_start;
        }
        else
            assert(false);

        if (tail_indrct_size!=0)
        CUDA_SAFE_CALL(cudaMemcpyAsync( d_vertex_addr+(shard_id*shard_bucket_num+tail_indrct_start)*ASSOCIATIVITY,
                                   vertex_addr+(pred_metas[pred].indrct_hdr_start+tail_indrct_head+(pred_slice_id-end_main_slice_id-1)*shard_bucket_num)*ASSOCIATIVITY,
                                   sizeof(vertex_t)*tail_indrct_size*ASSOCIATIVITY,
                                   cudaMemcpyHostToDevice,
                                   stream_id ));
    }

    void load_edge_shard(int pred, int pred_slice_id, int shard_id, cudaStream_t stream_id)
    {
        int end_slice_id = pred_edge_shard_size[pred]-1;
        uint64_t tail_size = 0;
        if(pred_slice_id==end_slice_id)
            tail_size = (pred_metas[pred].edge_end-pred_metas[pred].edge_start)%shard_entry_num;
        else
            tail_size = shard_entry_num;
        CUDA_SAFE_CALL(cudaMemcpyAsync( d_edge_addr+shard_id*shard_entry_num,
                                   edge_addr+pred_metas[pred].edge_start+pred_slice_id*shard_entry_num,
                                   sizeof(edge_t)*tail_size,
                                   cudaMemcpyHostToDevice,
                                   stream_id ));

    }

    // for evaluation
    void reset() {
        // 重置bitmap，重置vertex_allocation[], edge_allocation[]
        int init_array[shard_num] = { 0 };
        vertex_bitmap.assign(&init_array[0],&init_array[0]+shard_num);
        edge_bitmap.assign(&init_array[0],&init_array[0]+shard_num);


        for (int i=0;i<pred_num;++i) {
            // Siyuan: vertex_len是main hdr和indirect hdr共需要的shard数量
            int vertex_len = pred_vertex_shard_size[i];
            for(int j=0;j<vertex_len;++j)
                vertex_allocation[i][j] = SHARD_ID_ERR;

            int edge_len = pred_edge_shard_size[i];
            for(int j=0;j<edge_len;++j)
                edge_allocation[i][j] = SHARD_ID_ERR;
        }
    }


    // Siyuan: 参数4要使用与pid_in_pattern不同的Stream
    void load_predicate(int pid_to_load, int pid_in_pattern, request_or_reply &req, cudaStream_t stream_id, bool preload)
    {
        cout << "load_predicate: pid=" << pid_to_load << ", step=" << req.step + 1
                    << ", nshards=" << pred_vertex_shard_size[pid_to_load] << endl;

        // pred_ver_shard_sz[pid]记录了这个pid需要的shard数量;
        // 对于pred的每一个逻辑shard，都要分配一个物理shard
        for(int i=0;i<pred_vertex_shard_size[pid_to_load];++i) {
            if (vertex_allocation[pid_to_load][i] != -1)
                continue;

            int shard_id = get_next_free_vertex_shard(req.preds, pid_to_load, pid_in_pattern);

            if (shard_id != SHARD_ID_ERR) {
                load_vertex_shard(pid_to_load, i, shard_id, stream_id);
                vertex_allocation[pid_to_load][i] = shard_id;
            } else {
                // only fail in non-preload situation, since we can continue loading data during processing time
                if (!preload) {
                    assert(pid_to_load == pid_in_pattern);
                    cout << "ERROR: fail to allocate vertex shard for pid " << pid_in_pattern << ", step="<< req.step << endl;;
                    assert(false);
                }
            }

        }

        for(int i=0;i<pred_edge_shard_size[pid_to_load];++i) {
            if (edge_allocation[pid_to_load][i] != -1)
                continue;

            int shard_id = get_next_free_edge_shard(req.preds, pid_to_load, pid_in_pattern);

            if (shard_id != SHARD_ID_ERR) {
                load_edge_shard(pid_to_load,i,shard_id,stream_id);
                edge_allocation[pid_to_load][i] = shard_id;
            } else {
                // only fail in non-preload situation, since we can continue loading data during processing time
                if (!preload) {
                    assert(pid_to_load == pid_in_pattern);
                    cout << "ERROR: fail to allocate edge shard for pid " << pid_in_pattern << ", step="<< req.step << endl;;
                    assert(false);
                }
            }
        }

    }

};


