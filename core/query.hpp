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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

using namespace std;
using namespace boost::archive;

enum var_type {
    known_var,
    unknown_var,
    const_var
};

enum comp_device {CPU_comp, GPU_comp};

// defined as constexpr due to switch-case
constexpr int var_pair(int t1, int t2) { return ((t1 << 4) | t2); }

class request_or_reply {

public:
    int id = -1;     // query id
    int pid = -1;    // parqnt query id
    int tid = 0;     // engine thread id (MT)

    // runtime state
    int step = 0;
    int col_num = 0;
    int row_num = 0;

    bool blind = false;
    bool dummy = false;  // for mix-workload test
    bool sub_req = false;

    comp_device comp_dev = CPU_comp;
    char* gpu_history_table_ptr = nullptr; // pointer to history table on GPU
    char* gpu_origin_buffer_head = nullptr;// to record origin buffer head for sub request
    int gpu_history_table_size = 0;

    int64_t local_var = 0;
    vector<int64_t> cmd_chains; // N * (subject, predicat, direction, object)
    vector<int> result_table;
    vector<int> preds;

    request_or_reply() { }
    request_or_reply(bool dummy) { this->dummy = dummy; }

    request_or_reply(vector<int64_t> _cc): cmd_chains(_cc) { }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & id;
        ar & pid;
        ar & tid;
        ar & step;
        ar & col_num;
        ar & row_num;
        ar & blind;
        ar & sub_req;
        ar & comp_dev;
        ar & gpu_history_table_size;
        ar & local_var;
        ar & cmd_chains;
        ar & result_table;
        ar & preds;
    }

    void clear_data() { result_table.clear(); }

    bool is_finished() { return (step * 4 >= cmd_chains.size()); }

    bool is_first_handler() { return (step==0); }
    bool is_last_handler() { return ((step+1) * 4 >= cmd_chains.size()); }

    bool is_request() { return (id == -1); }

    bool start_from_index() {
        /*
         * Wukong assumes that its planner will generate a dummy pattern to hint
         * the query should start from a certain index (i.e., predicate or type).
         * For example: ?X __PREDICATE__  ub:undergraduateDegreeFrom
         *
         * NOTE: the graph exploration does not must start from this index,
         * on the contrary, starts from another index would prune bindings MORE efficiently
         * For example, ?X P0 ?Y, ?X P1 ?Z, ...
         *
         * ?X __PREDICATE__ PO <- // start from index vertex P0
         * ?X P1 ?Z .             // then from ?X's edge with P1
         *
         */
        if (is_tpid(cmd_chains[0])) {
            assert(cmd_chains[1] == PREDICATE_ID || cmd_chains[1] == TYPE_ID);
            return true;
        }
        return false;
    }

    var_type variable_type(int64_t vid) {
        if (vid >= 0)
            return const_var;

        if ((- vid) > col_num)
            return unknown_var;
        else
            return known_var;
    }

    int64_t var2column(int64_t vid) {
        assert(vid < 0); // pattern variable
        return ((- vid) - 1);
    }

    void set_col_num(int n) { col_num = n; }

    int get_col_num() { return col_num; };

    int get_row_num() {
        if (col_num == 0) return 0;
        if(gpu_history_table_ptr!=nullptr)
            return gpu_history_table_size / col_num;
        return result_table.size() / col_num;
    }

    int64_t get_row_col(int r, int c) {
        return result_table[col_num * r + c];
    }

    void append_row_to(int r, vector<int> &updated_result_table) {
        for (int c = 0; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }
};

class request_template {

public:
    vector<int64_t> cmd_chains;

    // no serialize
    vector<int64_t> ptypes_pos;            // the locations of random-constants
    vector<string> ptypes_str;             // the Types of random-constants
    vector<vector<int>> ptypes_grp;  // the candidates for random-constants

    request_or_reply instantiate(int seed) {
        //request_or_reply request(cmd_chains);
        for (int i = 0; i < ptypes_pos.size(); i++) {
            cmd_chains[ptypes_pos[i]] =
                ptypes_grp[i][seed % ptypes_grp[i].size()];
        }
        return request_or_reply(cmd_chains);
    }
};
