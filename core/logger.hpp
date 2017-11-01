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

#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "timer.hpp"

using namespace std;
using namespace boost::archive;


class Logger {
private:
    struct req_stats {
        int query_type;
        uint64_t start_time = 0ull;
        uint64_t end_time = 0ull;

        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & query_type;
            ar & start_time;
            ar & end_time;
        }
    };

    uint64_t init_time = 0ull, done_time = 0ull;
    // key是reqid, value是这个query的类型和耗时
    unordered_map<int, req_stats> stats_map;

    // key是query_type, value是这种query的CDF data
    unordered_map<int, vector<uint64_t>> cdf_map;


    float thpt = 0.0;
    bool is_finish;
    int num_query_types;

public:
    void init(int ntypes = 1) {
        init_time = timer::get_usec();
        is_finish = false;
        stats_map.clear();
        cdf_map.clear();
        num_query_types = ntypes;
        for (int i = 0; i < num_query_types; ++i) {
            cdf_map.insert(make_pair(i, std::vector<uint64_t>()));
        }
    }

    void start_record(int reqid, int type) {
        stats_map[reqid].query_type = type;
        stats_map[reqid].start_time = timer::get_usec() - init_time;
    }

    void end_record(int reqid) {
        assert(reqid != -1);
        stats_map[reqid].end_time = timer::get_usec() - init_time;
    }

    void finish() {
        if (!is_finish) {
            done_time = timer::get_usec();
            // clear outstanding request before calculating throughput
            clear_flying_reqs();
            thpt = 1000.0 * stats_map.size() / (done_time - init_time);
            is_finish = true;
        }
    }

    void clear_flying_reqs() {
        for (auto it = stats_map.begin(); it != stats_map.end(); ) {
            if (it->second.end_time == 0ull)
                it = stats_map.erase(it);
            else
                it++;
        }
    }

    // Siyuan: 这个merge有没有问题？
    void merge(Logger &other) {
        for (auto s : other.stats_map)
            stats_map[s.first] = s.second;
        thpt += other.thpt;
    }

    void print_thpt() {
        cout << "Time elapsed: "<< (done_time - init_time) / 1000 << "ms, " << "Throughput: " << thpt << "K queries/sec" << endl;
    }

    void print_latency(int cnt = 1) {
        cout << "(average) latency: " << ((done_time - init_time) / cnt) << " usec" << endl;
    }

    void aggregate() {
        for (auto s : stats_map) {
            cdf_map[s.second.query_type].push_back(s.second.end_time - s.second.start_time);
        }

        // sort
        for (int i = 0; i < num_query_types; ++i) {
            if (!cdf_map[i].empty())
                sort(cdf_map[i].begin(), cdf_map[i].end());
        }
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
        // Siyuan: change for debuging
        int print_rate = 100; //stats_map.size() > 100 ? 100 : stats_map.size();
        assert(is_finish);
        aggregate();


        cout << "Per-query CDF graph" << endl;
        int cnt, query_type, query_cnt = 0;
        map<int, vector<uint64_t>> cdf_res;
        for (auto e : cdf_map) {
            query_type = e.first;
            vector<uint64_t> &cdf = e.second;
            if (cdf.empty())
                continue;
            query_cnt++;
            cnt = 0;
            cout << "Query: " << query_type + 1 << ", size: " << cdf.size() << endl;
            cdf_res[query_type] = std::vector<uint64_t>();
            if (global_print_cdf) {
                for (int i = 0; i < cdf.size() && cnt < print_rate; i++) {
                    if ((i + 1) % (cdf.size() / print_rate) == 0) {
                        cnt++;
                        if (cnt != print_rate) {
                            cout << cdf[i] << "\t";
                            cdf_res[query_type].push_back(cdf[i]);
                        }
                        else {
                            cout << cdf[cdf.size() - 1] << "\t";
                            cdf_res[query_type].push_back(cdf[cdf.size() - 1]);
                        }

                        if (cnt % 5 == 0) cout << endl;
                    }
                }
                cout << endl;
            }
        }

        if (!global_print_cdf)
            return;

        cout << "CDF Res: " << endl;
        cout << "P ";
        for (int i = 1; i <= query_cnt; ++i) {
            cout << "Q" << i << " ";
        }
        cout << endl;
        // first 20 rows
        // P = 5 * (row - 1) - 1, (row >= 2)
        int row, idx, p;
        for (row = 1, idx = 1; row <= 20; ++row)  {
            if (row == 1)
                cout << row << " ";
            else
                cout << 5 * (row - 1) << " ";
            for (int i = 0; i < query_cnt; ++i) {
                if (row == 1) {
                    cout << cdf_res[i][0] << " ";
                } else {
                    p = 5 * (row - 1) - 1;
                    cout << cdf_res[i][p] << " ";
                }
            }
            cout << endl;
        }
        // last 5 rows
        for (p = p+1; p < 100; p++) {
            cout << (p+1) << " ";
            for (int i = 0; i < query_cnt; ++i) {
                cout << cdf_res[i][p] << " ";
            }
            cout << endl;
        }

    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & stats_map;
        ar & thpt;
    }
};
