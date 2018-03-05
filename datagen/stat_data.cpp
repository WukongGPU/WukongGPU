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

#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <dirent.h>
#include <stdio.h>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;

int main(int argc, char *argv[])
{
    vector<string> files;
    if (argc < 2) {
        assert(false);
    }
    printf("folder: %s\n", argv[1]);

    string dname(argv[1]);

    // files located on a shared filesystem (e.g., NFS)
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        printf("ERORR: failed to open directory (%s)\n", dname.c_str());
        exit(-1);
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;

        string fname(dname + ent->d_name);
        // Assume the fnames of RDF data files (ID-format) start with 'id_'.
        /// TODO: move RDF data files and metadata files to different directories
        if (boost::starts_with(fname, dname + "id_"))
            files.push_back(fname);

    }

    sort(files.begin(), files.end());

    uint64_t num_triples = 0;
    set<uint64_t> subjects;
    set<uint64_t> objects;
    uint64_t s, p, o;
    for (auto f : files) {
        ifstream file(f.c_str());
        while (file >> s >> p >> o) {
            subjects.insert(s);
            objects.insert(o);

            num_triples ++;
        }
        file.close();

    }

    printf("#triples: %lu, #subjects: %lu, #objects: %lu\n",
            num_triples, subjects.size(), objects.size());

    return 0;
}
