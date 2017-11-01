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

/* NOTE: math will conflict with other lib; so it's named mymath */
class mymath {
public:
    // Siyuan: r对distribution的和取模后落在哪个区间里，就返回那个区间的下标
    uint64_t static get_distribution(int r, std::vector<int>& distribution) {
        int sum = 0;
        for (int i = 0; i < distribution.size(); i++)
            sum += distribution[i];

        assert(sum > 0);
        r = r % sum;
        for (int i = 0; i < distribution.size(); i++) {
            if (r < distribution[i])
                return i;
            r -= distribution[i];
        }
        assert(false);
    }

    inline uint64_t static round_up(uint64_t n, int m) {
        return ((n + m - 1) / m * m);
    }

    inline uint64_t static round_down(uint64_t n, int m) {
        return (n / m * m);
    }

    inline uint64_t static div_round_up(uint64_t n, int m) {
        return ((n + m - 1) / m);
    }

    inline int static hash_mod(uint64_t n, int m) {
        if (m == 0)
            assert(false);
        return n % m;
    }

    // TomasWang's 64 bit integer hash
    static uint64_t hash_u64(uint64_t key) {
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);
        return key;
    }

    static uint64_t inverse_hash_u64(uint64_t key) {
        uint64_t tmp;

        // Invert key = key + (key << 31)
        tmp = key - (key << 31);
        key = key - (tmp << 31);

        // Invert key = key ^ (key >> 28)
        tmp = key ^ key >> 28;
        key = key ^ tmp >> 28;

        // Invert key *= 21
        key *= 14933078535860113213u;

        // Invert key = key ^ (key >> 14)
        tmp = key ^ key >> 14;
        tmp = key ^ tmp >> 14;
        tmp = key ^ tmp >> 14;
        key = key ^ tmp >> 14;

        // Invert key *= 265
        key *= 15244667743933553977u;

        // Invert key = key ^ (key >> 24)
        tmp = key ^ key >> 24;
        key = key ^ tmp >> 24;

        // Invert key = (~key) + (key << 21)
        tmp = ~key;
        tmp = ~(key - (tmp << 21));
        tmp = ~(key - (tmp << 21));
        key = ~(key - (tmp << 21));

        return key;
    }

};

class mytuple {
    int static compare_tuple(int N, std::vector<int>& vec,
                             int i, std::vector<int>& vec2, int j) {
        // ture means less or equal
        for (int t = 0; t < N; t++) {
            if (vec[i * N + t] < vec2[j * N + t])
                return -1;

            if (vec[i * N + t] > vec2[j * N + t])
                return 1;
        }
        return 0;
    }

    inline void static swap_tuple(int N, std::vector<int> &vec, int i, int j) {
        for (int t = 0; t < N; t++)
            std::swap(vec[i * N + t], vec[j * N + t]);
    }

    void static qsort_tuple_recursive(int N, std::vector<int> &vec, int begin, int end) {
        if (begin + 1 >= end)
            return ;

        int middle = begin;
        for (int iter = begin + 1; iter < end; iter++) {
            if (compare_tuple(N, vec, iter, vec, begin) == -1 ) {
                middle++;
                swap_tuple(N, vec, iter, middle);
            }
        }

        swap_tuple(N, vec, begin, middle);
        qsort_tuple_recursive(N, vec, begin, middle);
        qsort_tuple_recursive(N, vec, middle + 1, end);
    }

    bool static binary_search_tuple_recursive(int N, std::vector<int> &vec,
            std::vector<int> &target,
            int begin, int end) {
        if (begin >= end)
            return false;

        int middle = (begin + end) / 2;
        int r = compare_tuple(N, target, 0, vec, middle);
        if (r == 0)
            return true;

        if (r < 0)
            return binary_search_tuple_recursive(N, vec, target, begin, middle);
        else
            return binary_search_tuple_recursive(N, vec, target, middle + 1, end);
    }


public:
    bool static binary_search_tuple(int N, std::vector<int> &vec,
                                    std::vector<int> &target) {
        binary_search_tuple_recursive(N, vec, target, 0, vec.size() / N);
    }

    void static qsort_tuple(int N, std::vector<int>& vec) {
        qsort_tuple_recursive(N, vec, 0, vec.size() / N);
    }
};
