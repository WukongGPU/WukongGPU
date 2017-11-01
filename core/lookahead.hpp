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
#include <tbb/concurrent_vector.h>
#include <algorithm>

using namespace std;


/**
 * A look-ahead table reflecting the cached predicates on GPU
 *
 * @thread-safe
 */
class LookAhead {
private:
    vector<int> preds;
    pthread_spinlock_t lock;


public:
    LookAhead() {
        pthread_spin_init(&lock, 0);
    }

    bool check(const request_or_reply &req, vector<int> &miss_preds) {
        pthread_spin_lock(&lock);

        std::set_difference(req.preds.begin(), req.preds.end(), preds.begin(), preds.end(),
                std::inserter(miss_preds, miss_preds.begin()));

        pthread_spin_unlock(&lock);
        return miss_preds.empty();
    }

    void update(const vector<int> &new_preds) {
        pthread_spin_lock(&lock);
        for (auto e : new_preds) {
            preds.push_back(e);
        }
        pthread_spin_unlock(&lock);
    }

};


