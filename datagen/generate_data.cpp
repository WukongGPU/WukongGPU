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

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>


/**
 * transfer str-format RDF data into id-format RDF data (triple rows)
 */


/**
 * A simple manual
 *  $g++ -std=c++11 generate_data.cpp -o generate_data
 *  $./generate_data lubm_raw_40 id_lubm_40
 */

/**
 * How to generate minimal_index file
 *  $grep "<http://www.University0.edu>" str_normal >> str_normal_minimal
 *  $grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
 *  ...
 */

using namespace std;

/* logical we split id-mapping table to normal-vertex and index-vertex table,
   but mapping all strings into the same id space. We reserve 2^NBITS_IDX ids
   for index vertices. */
enum { NBITS_IDX = 17 };

int
main(int argc, char** argv)
{
    unordered_map<string, int64_t> str_to_id;
    vector<string> normal_str;  // normal-vertex id table (vid)
    vector<string> index_str;   // index-vertex (i.e, predicate or type) id  table (p/tid)

    if (argc != 3) {
        printf("usage: ./generate_data src_dir dst_dir\n");
        return -1;
    }

    char *sdir_name = argv[1];
    char *ddir_name = argv[2];

    // create destination directory
    if (mkdir(ddir_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0) {
        cout << "Error: Creating dst_dir (" << ddir_name << ") failed." << endl;
        exit(-1);
    }

    // open source directory
    DIR *sdir = opendir(sdir_name);
    if (!sdir) {
        cout << "Error: Opening src_dir (" << sdir_name << ") failed." << endl;
        exit(-1);
    }

    // reserve t/pid[0] to predicate-index
    str_to_id["__PREDICATE__"] = 0;
    index_str.push_back("__PREDICATE__");

    // reserve t/pid[1] to type-index
    str_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;
    index_str.push_back("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

    // reserve the first two ids for the class of index vertex (i.e, predicate and type)
    int64_t next_index_id = 2;
    int64_t next_normal_id = 1 << NBITS_IDX; // reserve 2^NBITS_IDX ids for index vertices
    int count = 0;

    struct dirent *dent;
    while ((dent = readdir(sdir)) != NULL) {
        if (dent->d_name[0] == '.')
            continue;

        ifstream ifile((string(sdir_name) + "/" + string(dent->d_name)).c_str());
        ofstream ofile((string(ddir_name) + "/id_" + string(dent->d_name)).c_str());
        cout << "Process No." << ++count << " input file: " << dent->d_name << "." << endl;

        // str-format: subject predicate object .
        string subject, predicate, object, dot;
        // read (str-format) input file
        while (ifile >> subject >> predicate >> object >> dot) {
            // add a new normal vertex (i.e., vid)
            if (str_to_id.find(subject) == str_to_id.end()) {
                str_to_id[subject] = next_normal_id;
                next_normal_id ++;
                normal_str.push_back(subject);
            }

            // add a new (predicate) index vertex (i.e., pid)
            if (str_to_id.find(predicate) == str_to_id.end()) {
                str_to_id[predicate] = next_index_id;
                next_index_id ++;
                index_str.push_back(predicate);
            }

            // treat different types as individual indexes
            if (predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>") {
                // add a new (type) index vertex (i.e., tidx)
                if (str_to_id.find(object) == str_to_id.end()) {
                    str_to_id[object] = next_index_id;
                    next_index_id ++;
                    index_str.push_back(object);
                }
            } else {
                // add a new normal vertex (i.e., vid)
                if (str_to_id.find(object) == str_to_id.end()) {
                    str_to_id[object] = next_normal_id;
                    next_normal_id ++;
                    normal_str.push_back(object);
                }
            }

            // write (id-format) output file
            int64_t triple[3];
            triple[0] = str_to_id[subject];
            triple[1] = str_to_id[predicate];
            triple[2] = str_to_id[object];
            ofile << triple[0] << "\t" << triple[1] << "\t" << triple[2] << endl;
        }
    }
    closedir(sdir);

    /* build ID-mapping (str2id) table file for normal vertices */
    {
        ofstream f_normal((string(ddir_name) + "/str_normal").c_str());
        for (int64_t i = 0; i < normal_str.size(); i++)
            f_normal << normal_str[i] << "\t" << str_to_id[normal_str[i]] << endl;
    }

    /* build ID-mapping (str2id) table file for index vertices */
    {
        ofstream f_index((string(ddir_name) + "/str_index").c_str());
        for (int64_t i = 0; i < index_str.size(); i++)
            f_index << index_str[i] << "\t" << str_to_id[index_str[i]] << endl;
    }

    cout << "#total_vertex = " << str_to_id.size() << endl;
    cout << "#normal_vertex = " << normal_str.size() << endl;
    cout << "#index_vertex = " << index_str.size() << endl;

    return 0;
}
