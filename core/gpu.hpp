#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <cassert>
#include "query.hpp"
#include "gpu_config.h"
#include "gstore.hpp"
#include "rcache.hpp"
#include "shard_manager.hpp"

enum QueryHandlerTypes {
    INDEX_TO_UNKNOWN,
    KNOWN_TO_UNKNOWN,
    KNOWN_TO_CONST,
    KNOWN_TO_KNOWN,
};

static int QUERY_ID_EMTPY = -1;

// 把RCache，ShardManager都放到这里面来
// @singleton
class GPU {
private:
    char *hstry_buf;
    int _history_size;

    GStore *_gstore;
    RCache *_gcache;
    ShardManager *_shardmanager;

    std::atomic_int qid;

    GPU() { }

public:
    static GPU &instance() {
        static GPU gpu;
        return gpu;
    }

    void init(GStore *gstore, RCache *cache, ShardManager *shardmanager) {
        _gstore = gstore;
        _gcache = cache;
        _shardmanager = shardmanager;
        qid.store(QUERY_ID_EMTPY);
    }

    int query_id() const {
        return qid.load();
    }

    void set_query_id(int value) {
        bool ret = qid.compare_exchange_strong(QUERY_ID_EMTPY, value);
        assert(ret);
    }

    void clear_query_id(int old) {
        bool ret = qid.compare_exchange_strong(old, QUERY_ID_EMTPY);
        assert(ret);
    }

    char *history_inbuf() {
        return (char *)_gcache->d_result_table;
    }

    char *history_outbuf() {
        return (char *)_gcache->d_updated_result_table;
    }

    int history_size() const {
        return _history_size;
    }

    void set_history_size(int size) {
        _history_size = size;
    }

    char *load_history_data(int *host_history, size_t size) {
        // Siyuan: 暂时保留rcache里的d_result_table, d_updated_result_table
        // 这个old design，等测试完split query的接收和处理逻辑之后再重构掉。
        GPU_ASSERT(cudaMemcpy(_gcache->d_result_table,
                      host_history,
                      sizeof(*host_history) * size,
                      cudaMemcpyHostToDevice));

        return (char *)_gcache->d_result_table;
    }

    bool load_graph_data(int pred, int pid_in_pattern, request_or_reply &req, cudaStream_t stream_id, bool preload) {
        // not implement
        assert(false);

        return false;
    }

    void compute(request_or_reply &req, enum QueryHandlerTypes handler_type) {
        assert(false);
    }

};


void GPU_init(GStore *gstore, RCache *rcache, ShardManager *shardmanager) {
    GPU &gpu = GPU::instance();
    gpu.init(gstore, rcache, shardmanager);
}


