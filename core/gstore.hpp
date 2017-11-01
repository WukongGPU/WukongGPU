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

#include <stdint.h> //uint64_t
#include <vector>
#include <set>
#include <iostream>
#include <pthread.h>

#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include "config.hpp"
#include "rdma_resource.hpp"

#include "mymath.hpp"
#include "timer.hpp"
#include "unit.hpp"

#include "data_statistic.hpp"
#include "rdf_meta.hpp"

using namespace std;

struct triple_t {
    uint64_t s; // subject
    uint64_t p; // predicate
    uint64_t o; // object

    triple_t(): s(0), p(0), o(0) { }

    triple_t(uint64_t _s, uint64_t _p, uint64_t _o): s(_s), p(_p), o(_o) { }
};

struct edge_sort_by_spo {
    /* inline bool operator()(const triple_t &t1, const triple_t &t2) {
     *     if (t1.s < t2.s)
     *         return true;
     *     else if (t1.s == t2.s)
     *         if (t1.p < t2.p)
     *             return true;
     *         else if (t1.p == t2.p && t1.o < t2.o)
     *             return true;
     *     return false;
     * } */
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.p < t2.p)
            return true;
        else if (t1.p == t2.p)
            if (t1.s < t2.s)
                return true;
            else if (t1.s == t2.s && t1.o < t2.o)
                return true;
        return false;
    }
};

struct edge_sort_by_ops {
    /* inline bool operator()(const triple_t &t1, const triple_t &t2) {
     *     if (t1.o < t2.o)
     *         return true;
     *     else if (t1.o == t2.o)
     *         if (t1.p < t2.p)
     *             return true;
     *         else if ((t1.p == t2.p) && (t1.s < t2.s))
     *             return true;
     *     return false;
     * } */
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.p < t2.p)
            return true;
        else if (t1.p == t2.p)
            if (t1.o < t2.o)
                return true;
            else if ((t1.o == t2.o) && (t1.s < t2.s))
                return true;
        return false;
    }
};

enum { NBITS_DIR = 1 };
enum { NBITS_IDX = 17 }; // equal to the size of t/pid
enum { NBITS_VID = (64 - NBITS_IDX - NBITS_DIR) }; // 0: index vertex, ID: normal vertex

enum { PREDICATE_ID = 0, TYPE_ID = 1 }; // reserve two special index IDs
enum dir_t { IN, OUT, CORUN }; // direction: IN=0, OUT=1, and optimization hints

static inline bool is_tpid(int id) { return (id > 1) && (id < (1 << NBITS_IDX)); }
static inline bool is_idx(int vid) { return (vid == 0); }

/**
 * predicate-base key/value store
 * key: vid | t/pid | direction
 * value: v/t/pid list
 */
struct ikey_t {
uint64_t dir : NBITS_DIR; // direction
uint64_t pid : NBITS_IDX; // predicate
uint64_t vid : NBITS_VID; // vertex

    ikey_t(): vid(0), pid(0), dir(0) { }

    ikey_t(uint64_t v, uint64_t p, uint64_t d): vid(v), pid(p), dir(d) {
        assert((vid == v) && (dir == d) && (pid == p)); // no key truncate
    }

    bool operator == (const ikey_t &key) {
        if ((vid == key.vid) && (pid == key.pid) && (dir == key.dir))
            return true;
        return false;
    }

    bool operator != (const ikey_t &key) { return !(operator == (key)); }

    bool is_empty() { return ((vid == 0) && (pid == 0) && (dir == 0)); }

    void print() { cout << "[" << vid << "|" << pid << "|" << dir << "]" << endl; }

    uint64_t hash() {
        uint64_t r = 0;
        r += vid;
        r <<= NBITS_IDX;
        r += pid;
        r <<= NBITS_DIR;
        r += dir;
        return mymath::hash_u64(r); // the standard hash is too slow (i.e., std::hash<uint64_t>()(r))
    }
};

// 64-bit internal pointer (size < 256M and off off < 64GB)
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR = 36 };

/// TODO: add sid and edge type in future
struct iptr_t {
uint64_t size: NBITS_SIZE;
uint64_t off: NBITS_PTR;

    iptr_t(): size(0), off(0) { }

    iptr_t(uint64_t s, uint64_t o): size(s), off(o) {
        // no truncated
        assert ((size == s) && (off == o));
    }

    bool operator == (const iptr_t &ptr) {
        if ((size == ptr.size) && (off == ptr.off))
            return true;
        return false;
    }

    bool operator != (const iptr_t &ptr) {
        return !(operator == (ptr));
    }
};

// 128-bit vertex (key)
struct vertex_t {
    ikey_t key; // 64-bit: vertex | predicate | direction
    iptr_t ptr; // 64-bit: size | offset
};

// 32-bit edge (value)
struct edge_t {
    uint32_t val;  // vertex ID
};


class GStore {
private:
    class RDMA_Cache {
        struct Item {
            pthread_spinlock_t lock;
            vertex_t v;
            Item() {
                pthread_spin_init(&lock, 0);
            }
        };

        static const int NUM_ITEMS = 100000;
        Item items[NUM_ITEMS];

    public:
        /// TODO: use more clever cache structure with lock-free implementation
        bool lookup(ikey_t key, vertex_t &ret) {
            if (!global_enable_caching)
                return false;

            int idx = key.hash() % NUM_ITEMS;
            bool found = false;
            pthread_spin_lock(&(items[idx].lock));
            if (items[idx].v.key == key) {
                ret = items[idx].v;
                found = true;
            }
            pthread_spin_unlock(&(items[idx].lock));
            return found;
        }

        void insert(vertex_t &v) {
            if (!global_enable_caching)
                return;

            int idx = v.key.hash() % NUM_ITEMS;
            pthread_spin_lock(&items[idx].lock);
            items[idx].v = v;
            pthread_spin_unlock(&items[idx].lock);
        }
    };

    static const int NUM_LOCKS = 1024;

    static const int MAIN_RATIO = 80; // the percentage of main headers (e.g., 80%)
    static const int ASSOCIATIVITY = 8;  // the associativity of slots in each bucket

    uint64_t sid;
    Mem *mem;


    vertex_t *vertices;
    edge_t *edges;


    // the size of slot is sizeof(vertex_t)
    // the size of entry is sizeof(edge_t)
    uint64_t num_slots;       // 1 bucket = ASSOCIATIVITY slots
    uint64_t num_buckets;     // main-header region (pre-allocated hash-table)
    uint64_t num_buckets_ext; // indirect-header region (dynamical allocation)
    uint64_t num_entries;     // entry region (dynamical allocation)
    int num_preds;

    // metadata for each predicate
    vector<pred_meta_t> pred_metas;

    // Siyuan: multiple engines will access global_pred_metas
    tbb::concurrent_unordered_map <int, vector<pred_meta_t> > global_pred_metas;
    // predicate metas for each machine
    // map<int, vector<pred_meta_t>> global_pred_metas;


    uint64_t *alloc_table[20];   // [tid][pid]
    pthread_mutex_t mutex;
    pthread_barrier_t barrier;
    pthread_spinlock_t lock;
    pthread_cond_t cv_pred_inserted;
    volatile int num_pred_parts;
    volatile int last_type_pid;
    set<int> pred_set;


    // allocated
    uint64_t last_ext;
    uint64_t last_entry;

    RDMA_Cache rdma_cache;

    pthread_spinlock_t entry_lock;
    pthread_spinlock_t bucket_ext_lock;
    pthread_spinlock_t bucket_locks[NUM_LOCKS]; // lock virtualization (see paper: vLokc CGO'13)


    uint64_t bucket_for_remote(ikey_t key, int dst_sid) {
        // assert(global_pred_metas.find(dst_sid) != global_pred_metas.end());
        vector<pred_meta_t> &remote_pred_metas = global_pred_metas[dst_sid];
        return remote_pred_metas[key.pid].main_hdr_start + key.hash() % remote_pred_metas[key.pid].partition_sz;
    }

    uint64_t bucket_for(ikey_t key) {
        // the smallest pid is 1
        //return (key.pid - 1) * partition_szs[key.pid] + key.hash() % partition_szs[key.pid];
        return pred_metas[key.pid].main_hdr_start + key.hash() % pred_metas[key.pid].partition_sz;
    }

    // cluster chaining hash-table (see paper: DrTM SOSP'15)
    uint64_t insert_key(ikey_t key) {
        uint64_t bucket_id = bucket_for(key);
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;

        bool found = false;
        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and resue the last slot to store key
            for (uint64_t i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                //assert(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    key.print();
                    vertices[slot_id].key.print();
                    assert(false);
                }

                // insert to an empty slot
                if (vertices[slot_id].key.is_empty()) {
                    vertices[slot_id].key = key;
                    goto done;
                }
            }

            // whether the bucket_ext (indirect-header region) is used
            if (!vertices[slot_id].key.is_empty()) {
                slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY;
                continue; // continue and jump to next bucket
            }


            // allocate and link a new indirect header
            pthread_spin_lock(&bucket_ext_lock);
            assert(last_ext < num_buckets_ext);
            vertices[slot_id].key.vid = num_buckets + (last_ext++);
            pthread_spin_unlock(&bucket_ext_lock);

            slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY; // move to a new bucket_ext
            vertices[slot_id].key = key; // insert to the first slot
            goto done;
        }
done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        assert(slot_id < num_slots);
        assert(vertices[slot_id].key == key);
        return slot_id;
    }

    uint64_t sync_fetch_and_alloc_edges(uint64_t n) {
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += n;
        assert(last_entry < num_entries);
        pthread_spin_unlock(&entry_lock);
        return orig;
    }

    vertex_t get_vertex_remote(int tid, ikey_t key) {
        int dst_sid = mymath::hash_mod(key.vid, global_num_servers);
        uint64_t bucket_id = bucket_for_remote(key, dst_sid);
        vertex_t vert;


        if (rdma_cache.lookup(key, vert))
            return vert; // found

        char *buf = mem->buffer(tid);
        while (true) {
            uint64_t off = bucket_id * ASSOCIATIVITY * sizeof(vertex_t);
            uint64_t sz = ASSOCIATIVITY * sizeof(vertex_t);

            RDMA &rdma = RDMA::get_rdma();
            rdma.dev->RdmaRead(tid, dst_sid, buf, sz, off);
            vertex_t *verts = (vertex_t *)buf;
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                if (i < ASSOCIATIVITY - 1) {
                    if (verts[i].key == key) {
                        rdma_cache.insert(verts[i]);
                        return verts[i]; // found
                    }
                } else {
                    if (verts[i].key.is_empty())
                        return vertex_t(); // not found

                    bucket_id = verts[i].key.vid; // move to next bucket
                    break; // break for-loop
                }
            }
        }
    }

    vertex_t get_vertex_local(int tid, ikey_t key) {
        uint64_t bucket_id = bucket_for(key);
        while (true) {
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    //data part
                    if (vertices[slot_id].key == key) {
                        //we found it
                        return vertices[slot_id];
                    }
                } else {
                    if (vertices[slot_id].key.is_empty())
                        return vertex_t(); // not found
                    
                    bucket_id = vertices[slot_id].key.vid; // move to next bucket
                    break; // break for-loop
                }
            }
        }
    }

    edge_t *get_edges_remote(int tid, int64_t vid, int64_t d, int64_t pid, int *size) {
        int dst_sid = mymath::hash_mod(vid, global_num_servers);
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_remote(tid, key);

        if (v.key.is_empty()) {
            *size = 0;
            return NULL; // not found
        }

        char *buf = mem->buffer(tid);
        uint64_t off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        uint64_t sz = v.ptr.size * sizeof(edge_t);
        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, sz, off);
        edge_t *result_ptr = (edge_t *)buf;

        *size = v.ptr.size;
        return result_ptr;
    }

    edge_t *get_edges_local(int tid, int64_t vid, int64_t d, int64_t pid, int *size) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_local(tid, key);

        if (v.key.is_empty()) {
            *size = 0;
            return NULL;
        }

        *size = v.ptr.size;
        uint64_t off = v.ptr.off;
        //std::cout<<"off"<<off<<std::endl;
        //std::cout<<"&(edges[off])->val"<<&(edges[off]).val<<std::endl;

        //for(int i=10000000;i<10000000+43291;++i)
        //    std::cout<<"edges[i].val"<<edges[i].val<<";"<<std::endl;
        //std::cin.get();
        return &(edges[off]);
    }


    typedef tbb::concurrent_hash_map<int64_t, vector< int64_t>> tbb_hash_map;
    typedef tbb::concurrent_unordered_set<int64_t> tbb_unordered_set;

    tbb_hash_map pidx_in_map; // predicate-index (IN)
    tbb_hash_map pidx_out_map; // predicate-index (OUT)
    tbb_hash_map tidx_map; // type-index

    void insert_type_index_map(tbb_hash_map &map, dir_t d) {
        int64_t pid = 0;
        uint64_t off;
        for (auto const &e : map) {
            pid = e.first;
            pred_metas[pid].indrct_hdr_start = num_buckets + last_ext;

            uint64_t sz = e.second.size();
            off = sync_fetch_and_alloc_edges(sz);
            pred_metas[pid].edge_start = off;

            ikey_t key = ikey_t(0, pid, d);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : e.second)
                edges[off++].val = vid;

            record_meta_end(pid, off);
            last_type_pid = pid;
        }

    }


    void record_meta_end(int pid, uint64_t &off) {
        if (num_buckets + last_ext > pred_metas[pid].indrct_hdr_end)
            pred_metas[pid].indrct_hdr_end = num_buckets + last_ext;
        if (off > pred_metas[pid].edge_end)
            pred_metas[pid].edge_end = off;
    }

    void insert_index_map(tbb_hash_map &map, dir_t d, bool &first) {
        if (map.empty())
            return;

        // Siyuan: DBPSB多线程插入的时候map.size()会大于1
        assert(1 == map.size());

        int64_t pid = 0;
        uint64_t off;
        for (auto const &e : map) {
            pid = e.first;

            uint64_t sz = e.second.size();
            off = sync_fetch_and_alloc_edges(sz);

            ikey_t key = ikey_t(0, pid, d);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : e.second)
                edges[off++].val = vid;


            record_meta_end(pid, off);
            pred_set.insert(pid);

        }

        /* edge_start += (off - edge_start);
         * first = true; */
    }

#ifdef VERSATILE
    tbb_unordered_set p_set; // all of predicates
    tbb_unordered_set v_set; // all of vertices (subjects and objects)

    void insert_index_set(tbb_unordered_set &set, dir_t d) {
        uint64_t sz = set.size();
        uint64_t off = sync_fetch_and_alloc_edges(sz);

        ikey_t key = ikey_t(0, TYPE_ID, d);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &e : set)
            edges[off++].val = e;
    }
#endif

public:
    GStore() { }
    // encoding rules of gstore
    // subject/object (vid) >= 2^17, 2^17 > predicate/type (p/tid) > 2^1,
    // TYPE_ID = 1, PREDICATE_ID = 0, OUT = 1, IN = 0
    //
    // NORMAL key/value pair
    //   key = [vid |    predicate | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
    //   key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
    //   key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
    // INDEX key/value pair
    //   key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
    //   key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
    //   key = [  0 |      TYPE_ID |    OUT]  value = [vid0, vid1, ..]  i.e., all objects/subjects
    //   key = [  0 |      TYPE_ID |    OUT]  value = [vid0, vid1, ..]  i.e., all predicates
    // Empty key
    //   key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init

    // GStore: key (main-header and indirect-header region) | value (entry region)
    // The key (head region) is a cluster chaining hash-table (with associativity)
    // The value (entry region) is a varying-size array
    GStore(uint64_t sid, Mem *mem, int num_preds): sid(sid), mem(mem),
        num_preds(num_preds) {
        num_slots = ((uint64_t)global_num_keys_million) * 1000 * 1000;
        num_buckets = (uint64_t)((num_slots / ASSOCIATIVITY) * MAIN_RATIO / 100);
        //num_buckets_ext = (num_slots / ASSOCIATIVITY) / (KEY_RATIO + 1);
        num_buckets_ext = (num_slots / ASSOCIATIVITY) - num_buckets;

        vertices = (vertex_t *)(mem->kvstore());
        edges = (edge_t *)(mem->kvstore() + num_slots * sizeof(vertex_t));

        if (mem->kvstore_size() <= num_slots * sizeof(vertex_t)) {
            cout << "ERROR: " << global_memstore_size_gb
                 << "GB memory store is not enough to store hash table with "
                 << global_num_keys_million << "M keys" << std::endl;
            assert(false);
        }

        num_entries = (mem->kvstore_size() - num_slots * sizeof(vertex_t)) / sizeof(edge_t);
        last_entry = 0;

        pthread_mutex_init(&mutex, 0);
        pthread_barrier_init(&barrier, 0, nthread_parallel_load);
        pthread_spin_init(&lock, 0);
        pthread_cond_init(&cv_pred_inserted, 0);
        num_pred_parts = 0;

        pthread_spin_init(&entry_lock, 0);
        pthread_spin_init(&bucket_ext_lock, 0);
        for (int i = 0; i < NUM_LOCKS; i++)
            pthread_spin_init(&bucket_locks[i], 0);
    }

    // TODO: 在这里初始化partition_szs，统计各个predicate的数据量，
    // 按比例分配num_buckets给各个predicate。
    // 前提：nthread_parallel_load == 1
    void init(const vector<vector<triple_t>> &triple_spo, const vector<vector<triple_t>> &triple_ops) {
        vector<uint64_t> partition_szs; // 最初存的是triple的数量
        // initialize
        for (int i = 0; i <= num_preds; ++i) {
            pred_metas.push_back(pred_meta_t());
            partition_szs.push_back(0);
            pred_metas[i].num_buckets = 0;
        }

        // init alloc_table
        for (int i = 0; i < nthread_parallel_load; ++i) {
            alloc_table[i] = new uint64_t[num_preds + 1];
            memset(alloc_table[i], 0, sizeof(uint64_t) * (num_preds + 1));
        }

        int pid, tid;
        uint64_t num_triples = 0, main_hdr_cnt = 0;
        // count triples for each predicate
        for (tid = 0; tid < nthread_parallel_load; tid++) {
            for (auto spo : triple_spo[tid]) {
                alloc_table[tid][spo.p]++;
                partition_szs[spo.p]++;
                num_triples++;
                if (is_tpid(spo.o)) {
                    partition_szs[spo.o]++;
                    num_triples++;
                }
            }

            for (auto ops: triple_ops[tid]) {
                if (is_tpid(ops.o))
                    continue;
                alloc_table[tid][ops.p]++;
            }
        }


        // allocate main headers
        for (int i = 1; i <= num_preds; ++i) {
            partition_szs[i] = (static_cast<double>(partition_szs[i]) / num_triples) * num_buckets;
            pred_metas[i].partition_sz = partition_szs[i];
            pred_metas[i].main_hdr_start = main_hdr_cnt;
            pred_metas[i].main_hdr_end = main_hdr_cnt + partition_szs[i];
            main_hdr_cnt += partition_szs[i];
        }


        // initiate keys
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t i = 0; i < num_slots; i++)
            vertices[i].key = ikey_t();
    }

    // skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    // because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_ops
    // has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    // Consequently, all TYPE triples are aggregated at the beggining of triple_ops
    void insert_normal(int tid, vector<triple_t> &spo, vector<triple_t> &ops) {
        set<int> spo_pid_set;
        set<int> ops_pid_set;

        for (auto triple : spo) {
            spo_pid_set.insert(triple.p);
        }

        for (auto triple : ops) {
            ops_pid_set.insert(triple.p);
        }

        printf("[thread %d] pid in spo [Server %d]:\n", tid, sid);
        for (auto pid : spo_pid_set) {
            printf("%d ", pid);
        }
        printf("\n");

        printf("[thread %d] pid in ops [Server %d]:\n", tid, sid);
        for (auto pid : ops_pid_set) {
            printf("%d ", pid);
        }
        printf("\n");

        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < ops.size() && is_tpid(ops[type_triples].o))
            type_triples++;

#ifdef VERSATILE
        // the number of separate combinations of subject/object and predicate
        uint64_t accum_predicate = 0;
#endif
        // allocate edges in entry region for triples
        /* uint64_t edge_start = sync_fetch_and_alloc_edges(spo.size() + ops.size() - type_triples);
         * uint64_t off = edge_start; */
        uint64_t edge_start, off;

        uint64_t si, sj, ei, ej;
        uint64_t slot_id;
        int pid, current_pid = 0;

        bool finished = false, first = true;


        for (si = 0, sj = type_triples; si < spo.size(); si = ei) {
            // predicate-based key (subject + predicate)
            ei = si + 1;

            pid = spo[si].p;

            if (current_pid == 0 || pid != current_pid) {
                off = sync_fetch_and_alloc_edges(alloc_table[tid][pid]);
                current_pid = pid;
            }

            while ((ei < spo.size())
                    && (spo[si].s == spo[ei].s)
                    && (spo[si].p == spo[ei].p)) {
                    ei++;
            }

#ifdef VERSATILE
            accum_predicate++;
#endif
            // insert vertex
            ikey_t key = ikey_t(spo[si].s, spo[si].p, OUT);
            slot_id = insert_key(key);
            iptr_t ptr = iptr_t(ei - si, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = si; i < ei; i++)
                edges[off++].val = spo[i].o;

            record_index_info(slot_id);

            // (subject + predicate) part of one predicate has been inserted,
            // insert its (object + predicate) part
            if (ei >= spo.size() || spo[si].p != spo[ei].p) {
                // for predicate type
                if (current_pid == TYPE_ID) {
                    pthread_mutex_lock(&mutex);
                    record_meta_end(key.pid, off);
                    pthread_mutex_unlock(&mutex);
                } else {
                    for ( ; sj < ops.size(); sj = ej) {
                        ej = sj + 1;

                        // sometimes a predicate(e.g telephone) only in vector spo
                        // but not in vector ops
                        if (ops[sj].p != current_pid)
                            break;  // break for

                        while ((ej < ops.size())
                                && (ops[sj].o == ops[ej].o)
                                && (ops[sj].p == ops[ej].p)) { ej++; }
#ifdef VERSATILE
                        accum_predicate++;
#endif

                        // insert vertex
                        ikey_t key = ikey_t(ops[sj].o, ops[sj].p, IN);
                        slot_id = insert_key(key);
                        iptr_t ptr = iptr_t(ej - sj, off);
                        vertices[slot_id].ptr = ptr;

                        // insert edges
                        for (uint64_t i = sj; i < ej; i++)
                            edges[off++].val = ops[i].s;

                        record_index_info(slot_id);

                        // check whether this predicate is finished
                        if (ops[sj].p != ops[ej].p) {
                            sj = ej;
                            break;
                        }
                    }
                }

                pthread_mutex_lock(&mutex);
                ++num_pred_parts;

                if (tid == 0) { // master
                    // wait until all worker finishes its part
                    while (num_pred_parts < nthread_parallel_load)
                        pthread_cond_wait(&cv_pred_inserted, &mutex);

                    insert_type_index_map(tidx_map, IN);
                    // insert_type_index_vec(tidx_vec, IN);
                    insert_index_map(pidx_in_map, IN, first);
                    insert_index_map(pidx_out_map, OUT, first);

                    // tbb_vec().swap(tidx_vec);
                    tbb_hash_map().swap(tidx_map);
                    tbb_hash_map().swap(pidx_in_map);
                    tbb_hash_map().swap(pidx_out_map);

                    num_pred_parts = 0;   // reset
                    pthread_mutex_unlock(&mutex);

                } else {    // slaves
                    if (num_pred_parts == nthread_parallel_load) {
                        pthread_cond_signal(&cv_pred_inserted);
                    }

                    pthread_mutex_unlock(&mutex);
                }

                // wait for all threads finish inserting a predicate
                pthread_barrier_wait(&barrier);


            }   // if


        }   // for

    }

    void record_index_info(uint64_t slot_id) {

        int64_t vid = vertices[slot_id].key.vid;
        int64_t pid = vertices[slot_id].key.pid;

        uint64_t sz = vertices[slot_id].ptr.size;
        uint64_t off = vertices[slot_id].ptr.off;

        if (vertices[slot_id].key.dir == IN) {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
                assert(false); // (IN) type triples should be skipped
            } else { // predicate-index (OUT) vid
                tbb_hash_map::accessor a;
                pidx_out_map.insert(a, pid);
                a->second.push_back(vid);
            }
        } else {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
                // type-index (IN) vid
                for (uint64_t e = 0; e < sz; e++) {
                    tbb_hash_map::accessor a;
                    tidx_map.insert(a, edges[off + e].val);
                    a->second.push_back(vid);
                    // tidx_vec[edges[off + e].val].push_back(vid);
                }
            } else { // predicate-index (IN) vid

                tbb_hash_map::accessor a;
                pidx_in_map.insert(a, pid);
                a->second.push_back(vid);

            }
        }
    }

    // insert P-index and T-index vertices
    void insert_index() {
        uint64_t t1 = timer::get_usec();

        // scan raw data to generate index data in parallel
        #pragma omp parallel for num_threads(global_num_engines)
        for (int bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) continue;

                int64_t vid = vertices[slot_id].key.vid;
                int64_t pid = vertices[slot_id].key.pid;

                uint64_t sz = vertices[slot_id].ptr.size;
                uint64_t off = vertices[slot_id].ptr.off;

                if (vertices[slot_id].key.dir == IN) {
                    if (pid == PREDICATE_ID) {
#ifdef VERSATILE
                        v_set.insert(vid);
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val);
#endif
                    } else if (pid == TYPE_ID) {
                        assert(false); // (IN) type triples should be skipped
                    } else { // predicate-index (OUT) vid
                        tbb_hash_map::accessor a;
                        pidx_out_map.insert(a, pid);
                        a->second.push_back(vid);
                    }
                } else {
                    if (pid == PREDICATE_ID) {
#ifdef VERSATILE
                        v_set.insert(vid);
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val);
#endif
                    } else if (pid == TYPE_ID) {
                        // type-index (IN) vid
                        for (uint64_t e = 0; e < sz; e++) {
                            tbb_hash_map::accessor a;
                            tidx_map.insert(a, edges[off + e].val);
                            a->second.push_back(vid);
                        }
                    } else { // predicate-index (IN) vid
                        tbb_hash_map::accessor a;
                        pidx_in_map.insert(a, pid);
                        a->second.push_back(vid);
                    }
                }
            }
        }

        uint64_t t2 = timer::get_usec();
        cout << (t2 - t1) / 1000 << " ms for (parallel) prepare index info" << endl;

        cout << "tidx_map: size: " << tidx_map.size() << endl;
        cout << "pidx_in_map: size: " << pidx_in_map.size() << endl;
        cout << "pidx_out_map: size: " << pidx_out_map.size() << endl;


        // add type/predicate index vertices
        /* insert_index_map(tidx_map, IN);
         * insert_index_map(pidx_in_map, IN);
         * insert_index_map(pidx_out_map, OUT); */

        tbb_hash_map().swap(pidx_in_map);
        tbb_hash_map().swap(pidx_out_map);
        tbb_hash_map().swap(tidx_map);

#ifdef VERSATILE
        insert_index_set(v_set, IN);
        insert_index_set(p_set, OUT);

        tbb_unordered_set().swap(v_set);
        tbb_unordered_set().swap(p_set);
#endif

        uint64_t t3 = timer::get_usec();
        cout << (t3 - t2) / 1000 << " ms for insert index data into gstore" << endl;
    }

    // TODO: 填充非type的predicate的edge_start, indrct_hdr_start
    void finish_pred_metas() {
        int pid, prev_pid;
        // type
        pred_metas[1].indrct_hdr_start = num_buckets;


        set<int>::iterator it = pred_set.begin();
        pid = *it;

        pred_metas[pid].edge_start = pred_metas[last_type_pid].edge_end;
        pred_metas[pid].indrct_hdr_start = pred_metas[last_type_pid].indrct_hdr_end;

        it++;
        for (prev_pid = pid; it != pred_set.end(); ++it) {
            pid = *it;

            pred_metas[pid].edge_start = pred_metas[prev_pid].edge_end;
            pred_metas[pid].indrct_hdr_start = pred_metas[prev_pid].indrct_hdr_end;

            prev_pid = pid;
        }
    }

    void print_pred_metas() {
        for (int pid = 1; pid < pred_metas.size(); ++pid) {
            printf("pid=%d, edge_start=%ld, edge_end=%ld, ind_hdr_start=%ld, ind_hdr_end=%ld\n", pid,
                    pred_metas[pid].edge_start, pred_metas[pid].edge_end,
                    pred_metas[pid].indrct_hdr_start, pred_metas[pid].indrct_hdr_end);
        }
    }

    // 统计predicate实际占用了多少个buckets
    void stat_predicates() {
        for (int bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY;

            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) continue;

                int64_t vid = vertices[slot_id].key.vid;
                int64_t pid = vertices[slot_id].key.pid;
                uint64_t off = vertices[slot_id].ptr.off;

                if (i == 0)
                    pred_metas[pid].num_buckets++;
            }
        }
    }



    // prepare data for planner
    void generate_statistic(data_statistic& statistic) {
        for (int bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) continue;

                int64_t vid = vertices[slot_id].key.vid;
                int64_t pid = vertices[slot_id].key.pid;
                uint64_t off = vertices[slot_id].ptr.off;

                if (pid == PREDICATE_ID) continue; // skip for index vertex

                unordered_map<int, int>& ptcount = statistic.predicate_to_triple;
                unordered_map<int, int>& pscount = statistic.predicate_to_subject;
                unordered_map<int, int>& pocount = statistic.predicate_to_object;
                unordered_map<int, int>& tyscount = statistic.type_to_subject;
                unordered_map<int, vector<direct_p> >& ipcount = statistic.id_to_predicate;

                if (vertices[slot_id].key.dir == IN) {
                    uint64_t sz = vertices[slot_id].ptr.size;
                    // triples only count from one direction
                    if (ptcount.find(pid) == ptcount.end()) {
                        ptcount[pid] = sz;
                    }
                    else {
                        ptcount[pid] += sz;
                    }
                    // count objects
                    if (pocount.find(pid) == pocount.end()) {
                        pocount[pid] = 1;
                    }
                    else {
                        pocount[pid] ++;
                    }
                    // count in predicates for specific id
                    ipcount[vid].push_back(direct_p(IN, pid));
                }
                else {
                    // count subjects
                    if (pscount.find(pid) == pscount.end()) {
                        pscount[pid] = 1;
                    }
                    else {
                        pscount[pid]++;
                    }
                    // count out predicates for specific id
                    ipcount[vid].push_back(direct_p(OUT, pid));
                    // count type predicate
                    if (pid == TYPE_ID) {
                        uint64_t sz = vertices[slot_id].ptr.size;
                        uint64_t off = vertices[slot_id].ptr.off;
                        for (uint64_t j = 0; j < sz; j++) {
                            //src may belongs to multiple types
                            uint64_t obid = edges[off + j].val;
                            if (tyscount.find(obid) == tyscount.end()) {
                                tyscount[obid] = 1;
                            }
                            else {
                                tyscount[obid] ++;
                            }
                            if (pscount.find(obid) == pscount.end()) {
                                pscount[obid] = 1;
                            } else {
                                pscount[obid] ++;
                            }
                            ipcount[vid].push_back(direct_p(OUT, obid));
                        }
                    }
                }
            }
        }

        //cout<<"sizeof predicate_to_triple = "<<statistic.predicate_to_triple.size()<<endl;
        //cout<<"sizeof predicate_to_subject = "<<statistic.predicate_to_subject.size()<<endl;
        //cout<<"sizeof predicate_to_object = "<<statistic.predicate_to_object.size()<<endl;
        //cout<<"sizeof type_to_subject = "<<statistic.type_to_subject.size()<<endl;
        //cout<<"sizeof id_to_predicate = "<<statistic.id_to_predicate.size()<<endl;

        unordered_map<pair<int, int>, four_num, boost::hash<pair<int, int> > >& ppcount = statistic.correlation;

        // do statistic for correlation
        for (unordered_map<int, vector<direct_p> >::iterator it = statistic.id_to_predicate.begin();
                it != statistic.id_to_predicate.end(); it++ ) {
            int vid = it->first;
            vector<direct_p>& vec = it->second;
            for (int i = 0; i < vec.size(); i++) {
                for (int j = i + 1; j < vec.size(); j++) {
                    int p1, d1, p2, d2;
                    if (vec[i].p < vec[j].p) {
                        p1 = vec[i].p;
                        d1 = vec[i].dir;
                        p2 = vec[j].p;
                        d2 = vec[j].dir;
                    } else {
                        p1 = vec[j].p;
                        d1 = vec[j].dir;
                        p2 = vec[i].p;
                        d2 = vec[i].dir;
                    }
                    if (d1 == OUT && d2 == OUT) {
                        ppcount[make_pair(p1, p2)].out_out++;
                    }
                    if (d1 == OUT && d2 == IN) {
                        ppcount[make_pair(p1, p2)].out_in++;
                    }
                    if (d1 == IN && d2 == IN) {
                        ppcount[make_pair(p1, p2)].in_in++;
                    }
                    if (d1 == IN && d2 == OUT) {
                        ppcount[make_pair(p1, p2)].in_out++;
                    }
                }
            }
        }
        //cout << "sizeof correlation = " << statistic.correlation.size() << endl;
        cout << "INFO#" << sid << ": generating statistics is finished." << endl;
    }

    edge_t *get_edges_global(int tid, int64_t vid, int64_t d, int64_t pid, int *sz) {
        if (mymath::hash_mod(vid, global_num_servers) == sid)
            return get_edges_local(tid, vid, d, pid, sz);
        else
            return get_edges_remote(tid, vid, d, pid, sz);
    }

    edge_t *get_index_edges_local(int tid, int64_t pid, int64_t d, int *sz) {
        // predicate is not important, so we set it 0
        return get_edges_local(tid, 0, d, pid, sz);
    }

    vertex_t *get_vertices_ptr(){return vertices;}
    edge_t *get_edges_ptr(){return edges;}
    uint64_t get_num_slots(){return num_slots;}
    uint64_t get_num_buckets(){return num_buckets;}
    uint64_t get_last_entry(){return last_entry;}
    vector<pred_meta_t> get_pred_metas(){return pred_metas;}
    int get_num_preds() { return num_preds; }

    void set_global_pred_metas(tbb::concurrent_unordered_map <int, vector<pred_meta_t> > &global_pred_metas) {
        this->global_pred_metas = global_pred_metas;
    }

    // analysis and debuging
    void print_mem_usage() {
        uint64_t used_slots = 0;
        for (int x = 0; x < num_buckets; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
            }
        }

        cout << "main header: " << B2MiB(num_buckets * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB (" << num_buckets * ASSOCIATIVITY << " slots)" << endl;
        cout << "\tused: " << 100.0 * used_slots / (num_buckets * ASSOCIATIVITY)
             << " % (" << used_slots << " slots)" << endl;
        cout << "\tchain: " << 100.0 * num_buckets / (num_buckets * ASSOCIATIVITY)
             << " % (" << num_buckets << " slots)" << endl;

        used_slots = 0;
        for (int x = num_buckets; x < num_buckets + num_buckets_ext; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
            }
        }

        cout << "indirect header: " << B2MiB(num_buckets_ext * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB (" << num_buckets_ext * ASSOCIATIVITY << " slots)" << endl;
        cout << "\talloced: " << 100.0 * last_ext / num_buckets_ext
             << " % (" << last_ext << " buckets)" << endl;
        cout << "\tused: " << 100.0 * used_slots / (num_buckets_ext * ASSOCIATIVITY)
             << " % (" << used_slots << " slots)" << endl;

        cout << "entry: " << B2MiB(num_entries * sizeof(edge_t))
             << " MB (" << num_entries << " entries)" << endl;
        cout << "\tused: " << 100.0 * last_entry / num_entries
             << " % (" << last_entry << " entries)" << endl;

    }
};
