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
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */
#pragma once
#include <cstdint>
#include <cassert>

#define WK_CLINE 64

#define TID_LAST_WORKER (global_num_threads - 1)
#define TID_FIRST_WORKER (global_num_proxies)
#define TID_GPU_AGENT (global_num_proxies + global_num_engines)


static uint64_t inline floor(uint64_t original, uint64_t n) {
    assert(n != 0);
    return original - original % n;
}

static uint64_t inline ceil(uint64_t original, uint64_t n) {
    assert(n != 0);
    if (original % n == 0)
        return original;
    return original - original % n + n;
}

namespace GPU_Config {
    // vertex frame中有多少个buckets
    uint64_t vertex_frame_num_buckets = 0;
    // edge frame中有多少个entries
    uint64_t edge_frame_num_entries = 0;

    // GCache上vertex frame的总数
    int gcache_num_vertex_frames = 0;

    // GCache上edge frame的总数
    int gcache_num_edge_frames = 0;
};


class Statistic {
public:
    uint64_t gdr_data_sz;
    int gdr_times;
    int vertex_swap_times;
    int edge_swap_times;
    std::map<int, int> pred_swap_blocks;
    Statistic() {
        vertex_swap_times = 0;
        edge_swap_times = 0;
        gdr_data_sz = gdr_times = 0;
    }

    void reset_swap_stats() {
        vertex_swap_times = 0;
        edge_swap_times = 0;
        pred_swap_blocks.clear();
    }

    void reset_gdr_stats() {
        gdr_data_sz = gdr_times = 0;
    }

};

Statistic global_stat;
