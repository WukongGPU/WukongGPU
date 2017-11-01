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

#define QUERY_TYPE "query"
#define ERROR "error"
struct CS_Request {
	string type;
	bool use_file;
	string content;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar &type;
		ar &use_file;
		ar &content;
	}
};

struct CS_Reply {
	string type;
	string content;
	int column;
	vector<int64_t> column_table;
	vector<int64_t> result_table;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar &type;
		ar &content;
		ar &column;
		ar &column_table;
		ar &result_table;
	}

	void print(int max_row = 10) {
		int size = result_table.size();
		int pos = 0;
		for (int r = 0; r < max_row; r++) {
			for (int c = 0; c < column; c++) {
				if (pos == size)return;
				cout << result_table[pos++] << "\t";
			}
			cout << endl;
		}
	}
};
