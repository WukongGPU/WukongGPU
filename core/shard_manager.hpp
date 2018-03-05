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
#include <cuda_runtime.h>
#include "rdf_meta.hpp"
#include "defines.hpp"
#include "query.hpp"
#include "gpu_config.h"

using namespace std;


#define SHARD_ID_ERR -1
#define MAX_NUM_PREDICATES 100000

extern Statistic global_stat;

struct vertex_t;
struct edge_t;

/**
 * A manager for allocating free shards to predicates
 *
 */
class ShardManager {
public:

    int pred_num = 0;
    vector<pred_meta_t> pred_metas;

    // free list
    std::list<int> free_key_blocks;
    std::list<int> free_value_blocks;
    uint16_t key_cache_used_info[MAX_NUM_PREDICATES]; // only record key blocks,  for debug
    uint16_t value_cache_used_info[MAX_NUM_PREDICATES]; // only record key blocks,  for debug
    std::bitset<MAX_NUM_PREDICATES> key_bset;
    std::list<int> pids_in_kcache;

    std::bitset<MAX_NUM_PREDICATES> value_bset;
    std::list<int> pids_in_vcache;




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

    ShardManager(vertex_t* d_v_a, edge_t* d_e_a, vertex_t* v_a, edge_t* e_a, uint64_t num_buckets, uint64_t num_entries, vector<pred_meta_t> p_metas) {
        d_vertex_addr = d_v_a;
        d_edge_addr = d_e_a;
        vertex_addr = v_a;
        edge_addr = e_a;

        total_bucket_num = num_buckets;
        total_entry_num = num_entries;
        shard_bucket_num = GPU_Config::vertex_frame_num_buckets;  // 每个bucket shard有多少个bucket
        shard_entry_num = GPU_Config::edge_frame_num_entries;    // 每个entry shard有多少个entry

        pred_metas = p_metas;
        pred_num = pred_metas.size();
        printf("[INFO] ShardManager: pred_num=%d; GCache: [#vertex_frames=%d, #edge_frames=%d, vertex_frame_sz=%dMB, entry_frame_sz=%dMB]\n",
                pred_num, GPU_Config::gcache_num_vertex_frames, GPU_Config::gcache_num_edge_frames,
                shard_bucket_num * ASSOCIATIVITY * sizeof(vertex_t) / (1024*1024),
                shard_entry_num * sizeof(edge_t) / (1024*1024) );

        //init bitmap
        // vertex_bitmap.assign(GPU_Config::gcache_num_vertex_frames, 0);
        // edge_bitmap.assign(GPU_Config::gcache_num_edge_frames, 0);
        for (int i = 0; i < GPU_Config::gcache_num_vertex_frames; ++i) {
            free_key_blocks.push_back(i);
        }
        for (int i = 0; i < GPU_Config::gcache_num_edge_frames; ++i) {
            free_value_blocks.push_back(i);
        }

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

            key_cache_used_info[i] = 0;
            value_cache_used_info[i] = 0;
        }
    }

    void evict_kcache_blocks(vector<int>& conflicts, int pred_to_load, int pid_in_pattern, bool preload, int num_need_blks) {
        int pid;
        vector<int>::reverse_iterator rit;

        // #2 [normal case] Siyuan: 没有free shards了，那么evict cached predicate占用的blocks
        for (auto it = pids_in_kcache.begin(); it != pids_in_kcache.end(); ) {
            int i = *it;
            if (i == pred_to_load || i == pid_in_pattern) {
                it++;
                continue;
            }

            if (find(conflicts.begin(), conflicts.end(), i)!=conflicts.end()) {
                it++;
                continue;
            }

            // Siyuan: pred_vertex_shard_size记录的是这个predicate占用了多少个shards
            for(int j=0;j<pred_vertex_shard_size[i];++j) {
                // evict blocks of predicate i
                int block_id = vertex_allocation[i][j];
                if (block_id != SHARD_ID_ERR) {
                    vertex_allocation[i][j] = SHARD_ID_ERR;
                    key_cache_used_info[i] --;
                    if (key_cache_used_info[i] == 0) {
                        it = pids_in_kcache.erase(it);
                        // predicate i 在cache上已经没有blocks了
                        key_bset.set(i, false);
                    }
                    free_key_blocks.push_back(block_id);
                    global_stat.vertex_swap_times ++;
                    global_stat.pred_swap_blocks[pred_to_load] ++;

#ifdef WUKONG_DEBUG
                    printf("normal: evict vertex unit[pid=%d, unit=%d]\n", i, j);
#endif
                    if (free_key_blocks.size() >= num_need_blks) {
                        return;
                    }
                }
            }

        }


        // worst case
        for (rit = conflicts.rbegin(); rit != conflicts.rend(); rit++) {
            pid = *rit;
            // 跳过不在cache里的predicate
            if (!key_bset.test(pid))
                continue;

            if (pid == pred_to_load || pid == pid_in_pattern) {
                continue;
            }

            // Siyuan: pred_vertex_shard_size记录的是这个predicate占用了多少个shards
            for(int j=0;j<pred_vertex_shard_size[pid];++j) {
                // evict blocks of predicate pid
                int block_id = vertex_allocation[pid][j];
                if (block_id != SHARD_ID_ERR) {
                    vertex_allocation[pid][j] = SHARD_ID_ERR;
                    key_cache_used_info[pid] --;
                    if (key_cache_used_info[pid] == 0) {
                        for (auto it = pids_in_kcache.begin(); it != pids_in_kcache.end(); it++) {
                            if (*it == pid) {
                                pids_in_kcache.erase(it);
                                break;
                            }
                        }
                        key_bset.set(pid, false);
                    }
                    free_key_blocks.push_back(block_id);
                    global_stat.vertex_swap_times ++;

#ifdef WUKONG_DEBUG
                    global_stat.pred_swap_blocks[pred_to_load] ++;
                    printf("worst: evict vertex unit[pid=%d, unit=%d] \n", pid, j);
#endif
                    if (free_key_blocks.size() >= num_need_blks)
                        return;
                }
            }
        }

    }


    void evict_vcache_blocks(vector<int>& conflicts, int pred_to_load, int pid_in_pattern, bool preload, int num_need_blks) {
        // int block_id = SHARD_ID_ERR;
        vector<int>::reverse_iterator rit;
        // evict
        for (auto it = pids_in_vcache.begin(); it != pids_in_vcache.end(); ) {
            int i = *it;
            if (i == pred_to_load || i == pid_in_pattern) {
                it++;
                continue;
            }

            if (find(conflicts.begin(), conflicts.end(),i)!=conflicts.end()) {
                it++;
                continue;
            }

            for(int j=0;j<pred_edge_shard_size[i];++j) {
                int block_id = edge_allocation[i][j];
                if (block_id != SHARD_ID_ERR) {
                    edge_allocation[i][j] = SHARD_ID_ERR;
                    value_cache_used_info[i] --;
                    if (value_cache_used_info[i] == 0) {
                        it = pids_in_vcache.erase(it);
                        value_bset.set(i, false);
                    }

                    free_value_blocks.push_back(block_id);
                    global_stat.edge_swap_times ++;

#ifdef WUKONG_DEBUG
                    printf("normal: evict edge shard[pid=%d, unit=%d]\n", i, j);
#endif
                    if (free_value_blocks.size() >= num_need_blks) {
                        return;
                    }
                }
            }
        }

        //worst case: evict predicates related to query
        for(rit = conflicts.rbegin();  rit != conflicts.rend(); rit++) {
            int i = *rit;
            if (!value_bset.test(i))
                continue;

            if (i == pred_to_load || i == pid_in_pattern) {
                continue;
            }

            for(int j=0;j<pred_edge_shard_size[i];++j) {
                int block_id = edge_allocation[i][j];
                if (block_id != SHARD_ID_ERR) {
                    edge_allocation[i][j] = SHARD_ID_ERR;
                    value_cache_used_info[i] --;
                    if (value_cache_used_info[i] == 0) {
                        for (auto it = pids_in_vcache.begin(); it != pids_in_vcache.end(); it++) {
                            if (*it == i) {
                                pids_in_vcache.erase(it);
                                break;
                            }
                        }
                        value_bset.set(i, false);
                    }

                    free_value_blocks.push_back(block_id);
                    global_stat.edge_swap_times ++;
#ifdef WUKONG_DEBUG
                    printf("worst: evict edge shard[pid=%d, unit=%d]\n", i, j);
#endif
                    if (free_value_blocks.size() >= num_need_blks)
                        return;
                }
            }
        }

    }


    int get_next_free_edge_shard(vector<int> conflicts, int pred_to_load, int pred_to_execute) {
        int block_id = SHARD_ID_ERR;
        if (!free_value_blocks.empty()) {
            block_id = free_value_blocks.front();
            free_value_blocks.pop_front();
            goto done;
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
                    block_id = edge_allocation[i][j];
                    edge_allocation[i][j] = -1;
#ifdef WUKONG_DEBUG
                    printf("evict allocted edge shard[pid=%d, unit=%d]\n", i, j);
#endif
                    global_stat.edge_swap_times++;
                    goto done;
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
                    block_id = edge_allocation[i][j];
                    edge_allocation[i][j] = -1;
#ifdef WUKONG_DEBUG
                    printf("evict allocted edge shard[pid=%d, unit=%d]\n", i, j);
#endif
                    global_stat.edge_swap_times++;
                    goto done;
                }
        }

done:
        return block_id;
    }

        // Before processing the query, we should ensure the data of required predicates is loaded.
    bool check_pred_exist(int pred) {
        if (key_cache_used_info[pred] != pred_vertex_shard_size[pred])
            return false;

        if (value_cache_used_info[pred] != pred_edge_shard_size[pred])
            return false;

        return true;
    }

    vector<uint64_t> get_pred_vertex_headers(int pred) {
        assert(vertex_allocation[pred].size() == pred_vertex_shard_size[pred]);
        vector<uint64_t> headers;
        for(auto shard_id:vertex_allocation[pred])
            headers.push_back(shard_id*shard_bucket_num);
        return headers;
    }

    vector<uint64_t> get_pred_edge_headers(int pred) {
        vector<uint64_t> headers;
        for(auto shard_id:edge_allocation[pred])
            headers.push_back(shard_id*shard_entry_num);
        return headers;
    }

        // Siyuan: pred_slice_id 这是predicate的逻辑shard id，类似于inode的逻辑块号
        // shard_id是物理块号
    void load_vertex_shard(int pred, int pred_slice_id, int shard_id, cudaStream_t stream_id) {
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

    void load_edge_shard(int pred, int pred_slice_id, int shard_id, cudaStream_t stream_id) {
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

        for (int i=0;i<pred_num;++i) {
            // Siyuan: vertex_len是main hdr和indirect hdr共需要的shard数量
            int vertex_len = pred_vertex_shard_size[i];
            for(int j=0;j<vertex_len;++j) {
                int blk = vertex_allocation[i][j];
                if (blk != SHARD_ID_ERR) {
                    vertex_allocation[i][j] = SHARD_ID_ERR;
                    free_key_blocks.push_back(blk);
                }
            }

            int edge_len = pred_edge_shard_size[i];
            for(int j=0;j<edge_len;++j) {
                int blk = edge_allocation[i][j];
                if (blk != SHARD_ID_ERR) {
                    edge_allocation[i][j] = SHARD_ID_ERR;
                    free_value_blocks.push_back(blk);
                }
            }
        }
        assert(free_key_blocks.size() == GPU_Config::gcache_num_vertex_frames);
        assert(free_value_blocks.size() == GPU_Config::gcache_num_edge_frames);
    }


        // Siyuan: 参数4要使用与pid_in_pattern不同的Stream
    void load_predicate(int pid_to_load, int pid_in_pattern, request_or_reply &req, cudaStream_t stream_id, bool preload) {
        printf("\nload_predicate: pid=%d, step=%d, #key_blocks=%d, #value_blocks=%d, preload=%d\n",
                pid_to_load, req.step + 1, pred_vertex_shard_size[pid_to_load], pred_edge_shard_size[pid_to_load], preload);

        uint64_t num_need_blks = pred_vertex_shard_size[pid_to_load] - key_cache_used_info[pid_to_load];
        if (free_key_blocks.size() < num_need_blks) {
            evict_kcache_blocks(req.preds, pid_to_load, pid_in_pattern, preload, num_need_blks);
        }

        for(int i=0;i<pred_vertex_shard_size[pid_to_load];++i) {
            if (vertex_allocation[pid_to_load][i] != SHARD_ID_ERR)
                continue;

            // 如果走了一遍evict算法，free list还是空的，这种情况只能出现在preload的case里
            if (free_key_blocks.empty()) {
                if (preload)
                    break;

                printf("[ERROR] free list is empty in none-preload case!\n");
                assert(false);
            }

            int block_id = free_key_blocks.front();
            free_key_blocks.pop_front();

            if (block_id != SHARD_ID_ERR) {
                load_vertex_shard(pid_to_load, i, block_id, stream_id);
                vertex_allocation[pid_to_load][i] = block_id;

                // update metadata
                key_cache_used_info[pid_to_load] ++;
                if (!key_bset.test(pid_to_load)) {
                    key_bset.set(pid_to_load, true);
                    pids_in_kcache.push_back(pid_to_load);
                }
            } else {
                // only fail in non-preload situation, since we can continue loading data during processing time
                if (!preload) {
                    assert(pid_to_load == pid_in_pattern);
                    cout << "ERROR: fail to allocate shard for pid " << pid_in_pattern << ", step="<< req.step << endl;;
                    assert(false);
                } else {
                    break;
                }
            }
        }

        num_need_blks = pred_edge_shard_size[pid_to_load] - value_cache_used_info[pid_to_load];
        if (free_value_blocks.size() < num_need_blks) {
            evict_vcache_blocks(req.preds, pid_to_load, pid_in_pattern, preload, num_need_blks);
        }

        for(int i=0;i<pred_edge_shard_size[pid_to_load];++i) {
            if (edge_allocation[pid_to_load][i] != SHARD_ID_ERR)
                continue;

            if (free_value_blocks.empty()) {
                if (preload)
                    break;

                printf("[ERROR] free_value_blocks is empty in none-preload case!\n");
                assert(false);
            }

            int block_id = free_value_blocks.front();
            free_value_blocks.pop_front();

            if (block_id != SHARD_ID_ERR) {
                load_edge_shard(pid_to_load, i, block_id, stream_id);
                edge_allocation[pid_to_load][i] = block_id;

                // update metadata
                value_cache_used_info[pid_to_load] ++;
                if (!value_bset.test(pid_to_load)) {
                    value_bset.set(pid_to_load, true);
                    pids_in_vcache.push_back(pid_to_load);
                }
            } else {
                assert(false);
            }
        }

    }

};


