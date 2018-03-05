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

#include <map>
#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "timer.hpp"
#include "figures.hpp"
#include "gpu_engine.hpp"

using namespace std;
using namespace boost::archive;

extern std::vector<GPU_Engine *> gpu_engines;
extern Statistic global_stat;

class Logger {
private:
    struct req_stats {
        int query_type;
        uint64_t start_time = 0ull;
        uint64_t end_time = 0ull;
        int reply_cnt;

        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & query_type;
            ar & start_time;
            ar & end_time;
            ar & reply_cnt;
        }
    };

    uint64_t init_time = 0ull, done_time = 0ull;
    uint64_t last_time = 0ull, last_separator = 0ull;

    uint64_t interval = SEC(1);  // 1sec
    // key是reqid, value是这个query的类型和耗时
    unordered_map<int, req_stats> stats_map;

    // key: query_type, value: latency of query
    // ordered by query_type
    std::map<int, vector<uint64_t>> total_latency_map;

    // raw data
    uint64_t total_thpt_start_cnt = 0;
    uint64_t thpt_start_cnt[2] = {0};
    uint64_t thpt_start_time = 0ull;
    uint64_t start_send_cnt;
    uint64_t total_recv_cnt;
    double total_load;

    double thpts[2] = {0.0};
    double total_thpt;
    int last_recv_cnts[2] = {0};
    bool is_finish;
    bool is_aggregated;
    int nquery_types;
    int nlight_types;
    int nheavy_types;
    int recv_cnt;
    int pfactor;
    double load;

    // logger每跑完一次就填充一下这些Figure所需的数据
    Fig_Thpt_Latency *fig_thpt_latency;


public:

    void set_figure(Fig_Thpt_Latency *fig) {
        fig_thpt_latency = fig;
    }

    int get_query_cnt() const {
        return stats_map.size();
    }

    void init(int ntypes = 1) {
        init_time = timer::get_usec();
        last_time = 0ull;
        last_separator = 0ull;
        is_finish = is_aggregated = false;
        stats_map.clear();
        total_latency_map.clear();
        nquery_types = ntypes;
        recv_cnt = 0;
        load = 0.0;
        total_thpt_start_cnt = 0;
        total_recv_cnt = 0;
        start_send_cnt = 0;
        total_load = 0.0;
        total_thpt = 0.0;
        last_recv_cnts[0] = last_recv_cnts[1] = 0;
        for (int i = 0; i < nquery_types; ++i) {
            total_latency_map[i] = std::vector<uint64_t>();
        }
    }

    void init(int nlight, int nheavy, int pf) {
        nlight_types = nlight;
        nheavy_types = nheavy;
        pfactor = pf;
        init(nlight_types + nheavy_types);
    }


    void start_record(int reqid, int type, bool heavy = false) {
        stats_map[reqid].query_type = type;
        stats_map[reqid].start_time = timer::get_usec() - init_time;
        if (heavy) {
            stats_map[reqid].reply_cnt = global_num_servers;
        } else {
            stats_map[reqid].reply_cnt = 1;
        }
     }

    uint64_t end_record(int reqid) {
        assert(stats_map[reqid].reply_cnt > 0);
        if (--stats_map[reqid].reply_cnt == 0) {
            stats_map[reqid].end_time = timer::get_usec() - init_time;
            // number of queries that successfully served
            recv_cnt += 1;
        }

        return recv_cnt;
    }

    int get_recv_cnt() { return recv_cnt; }

    void finish() {
        if (!is_finish) {
            done_time = timer::get_usec();
            is_finish = true;
            // init count for merging
            total_recv_cnt = recv_cnt;
            total_load = load;
        }
    }


    // master proxy will merge test result of other loggers
    void merge(Logger &other) {
        for (auto s : other.stats_map)
            stats_map[s.first] = s.second;

        thpts[0] += other.thpts[0];
        thpts[1] += other.thpts[1];
        total_thpt += other.total_thpt;
        total_recv_cnt += other.total_recv_cnt;
        total_load += other.load;
    }

    void print_latency(int cnt = 1) {
        cout << "(average) latency: " << ((done_time - init_time) / cnt) << " usec" << endl;
        cout << "Swap times: vertex: " << global_stat.vertex_swap_times << ", edge: " << global_stat.edge_swap_times << endl;
        global_stat.reset_swap_stats();
    }

    void aggregate() {
        for (auto s : stats_map) {
            total_latency_map[s.second.query_type].push_back(s.second.end_time - s.second.start_time);
        }

        // sort
        for (int i = 0; i < nquery_types; ++i) {
            vector<uint64_t> &lats = total_latency_map[i];
            if (!lats.empty())
                sort(lats.begin(), lats.end());
        }

        is_aggregated = true;
    }


    void analyse() {
        uint64_t lats_median[2];
        uint64_t lats_99th[2];

        vector<uint64_t> light_lats;
        vector<uint64_t> heavy_lats;

        assert(is_aggregated);

        light_lats.reserve(total_recv_cnt);
        heavy_lats.reserve(200);

        // calc median & 99th percentile latency
        for (auto lat_map : total_latency_map) {
            int query_type = lat_map.first;
            //  BUG: gdb里看到这个lats数组里的元素是无序的
            vector<uint64_t> &lats = lat_map.second;
            if (query_type < nlight_types) {
                for (auto e : lats) {
                    light_lats.push_back(e);
                }
            } else {
                for (auto e : lats) {
                    heavy_lats.push_back(e);
                }
            }
        }

        sort(light_lats.begin(), light_lats.end());
        sort(heavy_lats.begin(), heavy_lats.end());

        int sizes[2] = {light_lats.size(), heavy_lats.size()};
        int idxs[2];

        idxs[0] = sizes[0] * 0.99;
        idxs[1] = sizes[1] * 0.99;

        lats_median[0] = light_lats[sizes[0] / 2];
        lats_median[1] = heavy_lats[sizes[1] / 2];

        lats_99th[0] = light_lats[ idxs[0] ];
        lats_99th[1] = heavy_lats[ idxs[1] ];

        fig_thpt_latency->add_data(pfactor, total_load, thpts, lats_median, lats_99th);
    }


    void print_data() {
        assert(fig_thpt_latency != nullptr);
        fig_thpt_latency->print();
    }


    void print_cdf() {
#if 0
        // print range throughput with certain interval
        vector<int> thpts;
        int print_interval = 200 * 1000; // 200ms

        for (auto s : stats_map) {
            int i = s.second.start_time / print_interval;
            if (thpts.size() <= i)
                thpts.resize(i + 1);
            thpts[i]++;
        }

        cout << "Range Throughput (K queries/sec)" << endl;
        for (int i = 0; i < thpts.size(); i++)
            cout << "[" << (print_interval * i) / 1000 << "ms ~ "
                 << print_interval * (i + 1) / 1000 << "ms)\t"
                 << (float)thpts[i] / (print_interval / 1000) << endl;
#endif
        // print CDF of query latency
        assert(is_finish);
        assert(is_aggregated);
        // aggregate(); Siyuan: called in console.hpp
        vector<double> cdf_rates = {0.01};

        for (int i = 1; i < 20; ++i) {
            cdf_rates.push_back(0.05 * i);
        }
        for (int i = 1; i <= 5; ++i) {
            cdf_rates.push_back(0.95 + i * 0.01);
        }
        assert(cdf_rates.size() == 25);

        cout << "Per-query CDF graph" << endl;
        int cnt, query_type;//, query_cnt = 0;
        map<int, vector<uint64_t>> cdf_res;

        // 从total_latency_map中选出25个点，作为最终的CDF图数据
        for (auto e : total_latency_map) {
            query_type = e.first;
            vector<uint64_t> &lats = e.second;
            // assert(lats.size() > cdf_rates.size());
            if (lats.empty())
                continue;

            cnt = 0;
            cout << "Query: " << query_type + 1 << ", size: " << lats.size() << endl;
            // result of CDF figure
            cdf_res[query_type] = std::vector<uint64_t>();
            if (global_print_cdf) {
                // 利用cdf_rates从lats中筛选出对应下标的点
                for (auto rate : cdf_rates) {
                    int idx = lats.size() * rate;
                    if (idx >= lats.size()) idx = lats.size() - 1;
                    cout << lats[idx] << "\t";
                    cdf_res[query_type].push_back(lats[idx]);
                    cnt++;
                    if (cnt % 5 == 0)   cout << endl;
                }

                assert(cdf_res[query_type].size() == 25);
                cout << endl;
            }
        }

        if (!global_print_cdf)
            return;

        cout << "CDF Res: " << endl;
        cout << "P ";
        for (int i = 1; i <= nquery_types; ++i) {
            cout << "Q" << i << " ";
        }
        cout << endl;

        // print cdf data
        int row, p;
        for (row = 1; row <= 25; ++row) {
            if (row == 1)
                cout << row << " ";
            else if (row <= 20)
                cout << 5 * (row - 1) << " ";
            else
                cout << 95 + (row - 20) << " ";

            for (int i = 0; i < nquery_types; ++i) {
                cout << cdf_res[i][row - 1] << " ";

            }
            cout << endl;
        }

    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & stats_map;
        ar & thpts;
        ar & total_thpt;
        ar & total_recv_cnt;
        ar & load;
    }

    /** Functions from upstream **/
    void start_calc_thpt(uint64_t recv_light_cnt, uint64_t recv_heavy_cnt, uint64_t send_cnt) {
        thpt_start_time = timer::get_usec();
        // cnt是为了记录开始计时 duration 的时候已经接收到了个query的responses
        thpt_start_cnt[0] = recv_light_cnt;
        thpt_start_cnt[1] = recv_heavy_cnt;
        total_thpt_start_cnt = recv_light_cnt + recv_heavy_cnt;

        start_send_cnt = send_cnt;
        printf(">>>> Start throughput evaluation!\n");
    }

    void end_calc_thpt(uint64_t recv_light_cnt, uint64_t recv_heavy_cnt, uint64_t send_cnt) {
        uint64_t duration = (timer::get_usec() - thpt_start_time);
        thpts[0] = 1000.0 * (recv_light_cnt - thpt_start_cnt[0]) / duration;
        thpts[1] = 1000.0 * (recv_heavy_cnt - thpt_start_cnt[1]) / duration;
        total_thpt = 1000.0 * ((recv_light_cnt + recv_heavy_cnt) - total_thpt_start_cnt) / duration;

        // load: X K queries/sec
        load = 1000.0 * (send_cnt - start_send_cnt) / duration;

        printf("<<<< End throughput evaluation!\n");
    }

    void print_thpt() {
        GPU_Engine *gpu_engine = gpu_engines[0];
        cout << "Summary: "<< (done_time - init_time) / 1000 << "ms, ";
        cout << "load: " << total_load << " K qs/sec" << ", ";
        cout << "Throughput: light:" << thpts[0] << "K queries/sec" << ", heavy: " << (int)(thpts[1] * 1000.0) << " queries/sec";
        cout << ", used_pending_queue: " << gpu_engine->get_used_pending_queue() << endl;
    }

    // print the throughput of a fixed interval
    void print_timely_thpt(uint64_t recv_light_cnt, uint64_t recv_heavy_cnt) {

        uint64_t now = timer::get_usec();
        // periodically print timely throughput
        if ((now - last_time) >= interval) {
            double cur_thpt[2];
            cur_thpt[0] = 1000.0 * (recv_light_cnt - last_recv_cnts[0]) / (now - last_time);
            cur_thpt[1] = 1000.0 * (recv_heavy_cnt - last_recv_cnts[1]) / (now - last_time);
            cout << "Timely Throughput: light: " << cur_thpt[0] << "K queries/sec, heavy: " << cur_thpt[1] * 1000.0 << " queries/sec" << endl;
            last_time = now;
            last_recv_cnts[0] = recv_light_cnt;
            last_recv_cnts[1] = recv_heavy_cnt;
        }

        // print separators per second
        if (now - last_separator > SEC(1)) {
            cout << "----[" << (now - init_time) / SEC(1) << "sec]----" << endl;
            last_separator = now;
        }
    }
};
