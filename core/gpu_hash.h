#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cmath>

#include "gpu_config.h"
#include "rdf_meta.hpp"

void generate_key_list_i2u(int *result_table,
                       int index_vertex,
                       int direction,
                       void *key_list,
                       int query_size,
                       cudaStream_t stream_id=0);

void generate_key_list_k2u(int *result_table,
                       int start,
                       int direction,
                       int predict,
                       int col_num,
                       void *key_list,
                       int query_size,
                       cudaStream_t stream_id=0);

void get_slot_id_list(void* d_vertex_addr,
                 void* d_key_list,
                 uint64_t* d_slot_id_list,
                 pred_meta_t* pred_metas,
                 uint64_t* vertex_headers,
                 uint64_t pred_vertex_shard_size,
                 int query_size,
                 cudaStream_t stream_id=0);

void get_edge_list(uint64_t *slot_id_list,
                    void *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size,
                    int query_size,
                    cudaStream_t stream_id=0);

void get_edge_list_k2k(uint64_t *slot_id_list,
                    void *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    int query_size,
                    void *edge_addr,
                    int *result_table,
                    int col_num,
                    int end,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size,
                    cudaStream_t stream_id=0);

void get_edge_list_k2c(uint64_t *slot_id_list,
                    void *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    int query_size,
                    void *edge_addr,
                    int end,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size,
                    cudaStream_t stream_id=0);

int update_result_table_i2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  void *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size,
                                  cudaStream_t stream_id=0
                                  );


int update_result_table_k2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  void *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size,
                                  int query_size,
                                  cudaStream_t stream_id=0);

int update_result_table_k2k(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  void *edge_addr,
                                  int end,
                                  int query_size,
                                  cudaStream_t stream_id=0);

void calc_prefix_sum(int* d_out_arr,
                     int* d_in_arr,
                     int query_size,
                     cudaStream_t stream_id=0);

void hash_dispatched_server_id(int *result_table,
                                  int *index_list,
                                  int start,
                                  int col_num,
                                  int num_sub_request,
                                  int query_size,
                                  cudaStream_t stream_id=0);

void update_result_table_sub(int *result_table,
                                  int *updated_result_table,
                                  int *mapping_list,
                                  int *server_id_list,
                                  int *prefix_sum_list,
                                  int column_num,
                                  int num_sub_request,
                                  int query_size,
                                  cudaStream_t stream_id=0);

void calc_dispatched_position(int *d_result_table,
                              int *d_mapping_list,
                              int *d_server_id_list,
                              int *d_server_sum_list,
                              int *gpu_sub_table_size_list,
                              int start,
                              int column_num,
                              int num_sub_request,
                              int query_size,
                              cudaStream_t stream_id=0);


