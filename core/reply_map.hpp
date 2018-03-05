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
#include <boost/unordered_map.hpp>
#include "query.hpp"

using namespace std;

// The map is used to colloect the replies of sub-queries in fork-join execution
class Reply_Map {
private:

    struct Item {
        int count;
        request_or_reply parent_request;
        request_or_reply merged_reply;
    };

    boost::unordered_map<int, Item> internal_item_map;

public:
    void put_parent_request(request_or_reply &r, int cnt) {
        Item data;
        data.count = cnt;
        data.parent_request = r;
        internal_item_map[r.id] = data;
    }

    void put_reply(request_or_reply &r) {
        int pid = r.pid;
        Item &data = internal_item_map[pid];

        vector<int> &result_table = data.merged_reply.result_table;
        data.count--;
        data.merged_reply.step = r.step;
        data.merged_reply.col_num = r.col_num;
        data.merged_reply.blind = r.blind;
        data.merged_reply.row_num += r.row_num;

        int new_size = result_table.size() + r.result_table.size();
        result_table.reserve(new_size);
        result_table.insert( result_table.end(), r.result_table.begin(), r.result_table.end());
    }

    bool is_ready(int pid) {
        return internal_item_map[pid].count == 0;
    }

    request_or_reply get_merged_reply(int pid) {
        request_or_reply r = internal_item_map[pid].parent_request;
        request_or_reply &merged_reply = internal_item_map[pid].merged_reply;

        r.step = merged_reply.step;
        r.col_num = merged_reply.col_num;
        r.blind = merged_reply.blind;
        r.row_num = merged_reply.row_num;

        r.result_table.swap(merged_reply.result_table);
        internal_item_map.erase(pid);
        return r;
    }
};
