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



