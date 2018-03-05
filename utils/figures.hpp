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
#include <map>
#include <stdio.h>

// interface
class Figure {

public:
    void virtual init() = 0;
    void virtual print() = 0;
    void virtual finish() = 0;
};



class Fig_Thpt_Latency : public Figure {

private:
    // <parallel_factor, median latency>
    std::map<int, uint64_t> lat_median_map[2];
    // <parallel_factor, 99th latency>
    std::map<int, uint64_t> lat_99th_map[2];

    // <parallel_factor, <load, thpt>>
    // Note: unif for throughput is "K queries/sec"
    std::map<int, std::pair<double, double>> thpts_map[2];
    bool is_finish;
public:
    void init() {
        is_finish = false;
    }
    /**
     *  parallel_factor  load  thpt  median  99th
     *
     */
    void print() {
        assert(lat_median_map[0].size() == lat_99th_map[0].size() &&
                lat_median_map[0].size() == thpts_map[0].size());
        int pf;
        double thpt, load, lat_median, lat_99th;

        printf("========== Fig. Throughput & Latency ==========\n");
        printf("Light query:\n");
        printf("pfactor\tload(K)\tthpt(K queries)\tmedian_latency\t99th_latency\n");
        for (auto pair : thpts_map[0]) {
            pf = pair.first;
            load = pair.second.first;
            thpt = pair.second.second;
            // ms
            lat_median = lat_median_map[0][pf] / 1000.0;
            lat_99th = lat_99th_map[0][pf] / 1000.0;
            printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\n", pf, load, thpt, lat_median, lat_99th);
        }

        printf("\n");
        printf("Heavy query:\n");
        printf("pfactor\tload(K)\tthpt(queries)\tmedian_latency\t99th_latency\n");
        for (auto pair : thpts_map[1]) {
            pf = pair.first;
            load = pair.second.first;
            thpt = pair.second.second;
            lat_median = lat_median_map[1][pf] / 1000.0;
            lat_99th = lat_99th_map[1][pf] / 1000.0;
            printf("%d\t%.2f\t%.1f\t%.2f\t%.2f\n", pf, load, thpt * 1000.0, lat_median, lat_99th);
        }
    }

    // load是一个proxy每秒钟发送的query个数
    void add_data(int pfactor, double load, double thpts[], uint64_t lats_median[], uint64_t lats_99th[]) {
        // light
        lat_median_map[0][pfactor] = lats_median[0];
        lat_99th_map[0][pfactor] = lats_99th[0];
        thpts_map[0][pfactor] = std::make_pair(load, thpts[0]);

        // heavy
        lat_median_map[1][pfactor] = lats_median[1];
        lat_99th_map[1][pfactor] = lats_99th[1];
        thpts_map[1][pfactor] = std::make_pair(load, thpts[1]);
    }

    void finish() {
        is_finish = true;
    }

};
