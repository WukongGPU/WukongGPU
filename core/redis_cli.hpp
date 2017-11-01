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

#include <hiredis.h>
#include <cstdlib>
#include <cassert>

#pragma once

class Redis_Cli {

private:
    redisContext *ctx;
    int sid;


public:
    Redis_Cli(int sid) : sid(sid) {
        struct timeval timeout = { 1, 500000 }; // 1.5 seconds
        ctx = redisConnectWithTimeout("meepo0", 8888, timeout);
        if (ctx == NULL || ctx->err) {
            if (ctx) {
                printf("Connection error: %s\n", ctx->errstr);
                redisFree(ctx);
            } else {
                printf("Connection error: can't allocate redis context\n");
            }
            exit(1);
        }
    }

    ~Redis_Cli() {
        redisFree(ctx);
    }

    int incr(int sid) {
        int ret;
        redisReply *reply;
        reply = (redisReply *)redisCommand(ctx, "INCR GPU_Engine[%d]", sid);

        assert(reply != NULL);
        ret = reply->integer;
        freeReplyObject(reply);
        return ret;
    }

    int decr(int sid) {
        int ret;
        redisReply *reply;
        reply = (redisReply *)redisCommand(ctx, "DECR GPU_Engine[%d]", sid);

        assert(reply != NULL);
        ret = reply->integer;
        freeReplyObject(reply);
        return ret;
    }

    void set(const char *key, uint64_t offset) {
        redisReply *reply;
        reply = (redisReply *)redisCommand(ctx, "SET %s %llu", key, offset);
        assert(reply != NULL);
        printf("SET %s: %llu\n", key, offset);
        freeReplyObject(reply);
    }

    uint64_t get(const char *key) {
        uint64_t ret;
        redisReply *reply;
        reply = (redisReply *)redisCommand(ctx, "GET %s", key);
        assert(reply != NULL);
        printf("GET %s: %s\n", key, reply->str);
        ret = atoi(reply->str);
        freeReplyObject(reply);
        return ret;
    }


};
