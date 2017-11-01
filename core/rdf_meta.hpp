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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>
using namespace std;
using namespace boost::archive;

// metadata of a RDF predicate
struct pred_meta_t {
    uint64_t partition_sz;  // allocated main headers (hash space)
    uint64_t main_hdr_start;
    uint64_t main_hdr_end;
    uint64_t indrct_hdr_start;
    uint64_t indrct_hdr_end;
    uint64_t num_buckets;   // sum of used main headers and indirect headers
    uint64_t edge_start;
    uint64_t edge_end;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & partition_sz;
        ar & main_hdr_start;
        ar & main_hdr_end;
        ar & indrct_hdr_start;
        ar & indrct_hdr_end;
        ar & num_buckets;
        ar & edge_start;
        ar & edge_end;
    }


};


class Pred_Metas_Msg {
public:
    int sid;
    vector<pred_meta_t> data;

    Pred_Metas_Msg() { }

    Pred_Metas_Msg(vector<pred_meta_t> data) {
        this->data = data;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & sid;
        ar & data;
    }

};

