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

#include <stdlib.h>

extern int global_num_servers;
extern int global_num_threads;

using namespace std;

class Coder {

private:
	int sid;    // server id
	int tid;    // thread id

	// Note that overflow of qid is innocent if there is no long-running
	// fork-join query. Because we use qid to recognize the owner sid
	// and tid, as well as collect the results of sub-queries.
	int qid;  // The ID of each (sub-)query

	unsigned int seed;

public:
	Coder(int sid, int tid): sid(sid), tid(tid) {
		qid = global_num_threads * sid + tid;
		seed = qid;
	}

	~Coder() { }

	unsigned int get_random() { return rand_r(&seed); }

	int get_and_inc_qid() {
		int _id = qid;
		qid += global_num_servers * global_num_threads;
		if (qid < 0) qid = global_num_threads * sid + tid; // reset
		return _id;
	}

	int sid_of(int qid) {
		return (qid % (global_num_servers * global_num_threads)) / global_num_threads;
	}

	int tid_of(int qid) {
		return qid % global_num_threads;
	}
};
