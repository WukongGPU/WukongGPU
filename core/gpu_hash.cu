#include "gpu_hash.h"

/////////////////////////////////////////////////////////
//
//
//                         Utils
//
//
/////////////////////////////////////////////////////////
enum { NBITS_DIR = 1 };
enum { NBITS_IDX = 17 }; // equal to the size of t/pid
enum { NBITS_VID = (64 - NBITS_IDX - NBITS_DIR) }; // 0: index vertex, ID: normal vertex


struct ikey_t {
uint64_t dir : NBITS_DIR; // direction
uint64_t pid : NBITS_IDX; // predicate
uint64_t vid : NBITS_VID; // vertex

    __host__ __device__
    ikey_t(): vid(0), pid(0), dir(0) { }

    __host__ __device__
    ikey_t(uint64_t v, uint64_t p, uint64_t d): vid(v), pid(p), dir(d) {
    }
};

// 64-bit internal pointer (size < 256M and off off < 64GB)
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR = 36 };

/// TODO: add sid and edge type in future
struct iptr_t {
uint64_t size: NBITS_SIZE;
uint64_t off: NBITS_PTR;

    __device__
    iptr_t(): size(0), off(0) { }

    __device__
    iptr_t(uint64_t s, uint64_t o): size(s), off(o) {
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

__device__
static uint64_t hash(ikey_t lkey)
{
    uint64_t r = 0;
    r += lkey.vid;
    r <<= NBITS_IDX;
    r += lkey.pid;
    r <<= NBITS_DIR;
    r += lkey.dir;

    uint64_t key = r;
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

__device__
bool compare_keys(ikey_t &k1, ikey_t &k2)
{
    if(k1.dir==k2.dir
            && k1.pid==k2.pid
            && k1.vid==k2.vid){
			    return true;
		}
    return false;
}

__device__
uint64_t map_location_on_shards(uint64_t offset, uint64_t *head_list, uint64_t shard_sz)
{
    return head_list[offset/shard_sz]+offset%shard_sz;
}



static uint64_t get_usec() {
    struct timespec tp;
    /* POSIX.1-2008: Applications should use the clock_gettime() function
       instead of the obsolescent gettimeofday() function. */
    /* NOTE: The clock_gettime() function is only available on Linux.
       The mach_absolute_time() function is an alternative on OSX. */
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return ((tp.tv_sec * 1000 * 1000) + (tp.tv_nsec / 1000));
}


/////////////////////////////////////////////////////////
//
//
//                   Query function
//
//
/////////////////////////////////////////////////////////


__global__
void d_generate_key_list_i2u(int *result_table,
                                int index_vertex,
                                int direction,
                                ikey_t *key_list,
                                int size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<size) {
        ikey_t r = ikey_t(0,index_vertex,direction);
        key_list[index] = r;
    }
}


void generate_key_list_i2u(int *result_table,
                       int index_vertex,
                       int direction,
                       void *key_list,
                       int query_size,
                       cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);
    d_generate_key_list_i2u << < dimGrid, dimBlock, 0, stream_id >> > (result_table, index_vertex, direction, (ikey_t*) key_list, query_size);
}

__global__
void d_generate_key_list_k2u(int *result_table,
                                int start,
                                int direction,
                                int predict,
                                int col_num,
                                ikey_t *key_list,
                                int size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<size) {
        int prev_id = result_table[index * col_num - start - 1];
        ikey_t r = ikey_t(prev_id,predict,direction);
        key_list[index] = r;
    }
}


void generate_key_list_k2u(int *result_table,
                       int start,
                       int direction,
                       int predict,
                       int col_num,
                       void *key_list,
                       int query_size,
                       cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);
    d_generate_key_list_k2u << < dimGrid, dimBlock, 0, stream_id  >> > (result_table, start, direction, predict, col_num, (ikey_t*) key_list, query_size);
}



__global__
void d_get_slot_id_list(vertex_t* d_vertex_addr,
                 ikey_t* d_key_list,
                 uint64_t* d_slot_id_list,
                 ikey_t empty_key,
                 pred_meta_t *pred_metas,
                 uint64_t* vertex_headers,
                 uint64_t pred_vertex_shard_size,
                 int query_size)

{

    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;

    if (index < query_size) {
        ikey_t key =  d_key_list[index];
        uint64_t bucket_id=map_location_on_shards(hash(key) % pred_metas[key.pid].partition_sz,
                                                  vertex_headers,
                                                  pred_vertex_shard_size);
        //int jump_count = 0;
        while(true){
            for(uint64_t i=0;i<ASSOCIATIVITY;i++){
                uint64_t slot_id=bucket_id*ASSOCIATIVITY+i;
                if(i<ASSOCIATIVITY-1){
                    //data part
                    if(compare_keys(d_vertex_addr[slot_id].key,d_key_list[index])){
                        //we found it
                        //d_slot_id_list[index] = jump_count;
                        d_slot_id_list[index] = slot_id;
                        //if(jump_count>0) printf("indrct found\n");
                        return;
                    }
                } else {
                    if(!compare_keys(d_vertex_addr[slot_id].key,empty_key)){
                        //next pointer
                        uint64_t next_bucket_id = d_vertex_addr[slot_id].key.vid-pred_metas[key.pid].indrct_hdr_start+pred_metas[key.pid].partition_sz;
                        bucket_id=map_location_on_shards(next_bucket_id,
                                                         vertex_headers,
                                                         pred_vertex_shard_size);
                        //jump_count += 1;
                        //break from for loop, will go to next bucket
                        break;
                    } else {
                        d_slot_id_list[index] = (uint64_t)(-1);
                        //printf("ALERT: cannot find slot for key %lu at index %d\n",hash(key) ,index);
                        return;
                    }
                }
            }
        }
    }

}

void get_slot_id_list(void* d_vertex_addr,
                 void* d_key_list,
                 uint64_t* d_slot_id_list,
                 pred_meta_t* pred_metas,
                 uint64_t* vertex_headers,
                 uint64_t pred_vertex_shard_size,
                 int query_size,
                 cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    ikey_t empty_key = ikey_t();

    d_get_slot_id_list<<<dimGrid, dimBlock, 0, stream_id >>>((vertex_t*)d_vertex_addr,
                                       (ikey_t*)d_key_list,
                                       d_slot_id_list,
                                       empty_key,
                                       pred_metas,
                                       vertex_headers,
                                       pred_vertex_shard_size,
                                       query_size);
}

__global__
void d_get_edge_list(uint64_t *slot_id_list,
                    vertex_t *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *off_list,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size,
                    int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<query_size)
    {
        uint64_t id = slot_id_list[index];
        iptr_t r = d_vertex_addr[id].ptr;
        //if (index<10)
        //printf("r.size:%d\n",r.size);
        index_list_mirror[index] = r.size;
        //off_list[index] = map_location_on_shards(r.off-pred_orin_edge_start,
        //                                         edge_headers,
        //                                         pred_edge_shard_size);
        off_list[index] = r.off-pred_orin_edge_start;
   }


}

void get_edge_list(uint64_t *slot_id_list,
                    void *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size,
                    int query_size,
                    cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_get_edge_list << < dimGrid, dimBlock, 0, stream_id  >> > (
                    slot_id_list,
                    (vertex_t*)d_vertex_addr,
                    index_list,
                    index_list_mirror,
                    ptr_list,
                    pred_orin_edge_start,
                    edge_headers,
                    pred_edge_shard_size,
                    query_size);

}



__global__
void d_get_edge_list_k2k(uint64_t *slot_id_list,
                    vertex_t *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    int query_size,
                    edge_t *edge_addr,
                    int *result_table,
                    int col_num,
                    int end,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<query_size)
    {
        uint64_t id = slot_id_list[index];
        iptr_t r = d_vertex_addr[id].ptr;

        index_list_mirror[index] = 0;

        int end_id = result_table[index * col_num - end - 1];
        ptr_list[index] = r.off-pred_orin_edge_start;
        for(int k=0;k<r.size;k++){
            uint64_t ptr = map_location_on_shards(r.off-pred_orin_edge_start+k,
                                                  edge_headers,
                                                  pred_edge_shard_size);

            if (edge_addr[ptr].val==end_id)
            {
                index_list_mirror[index] = 1;
                break;
            }
        }
   }


}

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
                    cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_get_edge_list_k2k << < dimGrid, dimBlock, 0, stream_id  >> > (
                    slot_id_list,
                    (vertex_t*)d_vertex_addr,
                    index_list,
                    index_list_mirror,
                    ptr_list,
                    query_size,
                    (edge_t*)edge_addr,
                    result_table,
                    col_num,
                    end,
                    pred_orin_edge_start,
                    edge_headers,
                    pred_edge_shard_size);

}


__global__
void d_get_edge_list_k2c(uint64_t *slot_id_list,
                    vertex_t *d_vertex_addr,
                    int *index_list,
                    int *index_list_mirror,
                    uint64_t *ptr_list,
                    int query_size,
                    edge_t *edge_addr,
                    int end,
                    uint64_t pred_orin_edge_start,
                    uint64_t* edge_headers,
                    uint64_t pred_edge_shard_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<query_size)
    {
        uint64_t id = slot_id_list[index];
        iptr_t r = d_vertex_addr[id].ptr;

        index_list_mirror[index] = 0;
        ptr_list[index] =r.off-pred_orin_edge_start;
        for(int k=0;k<r.size;k++){
            uint64_t ptr = map_location_on_shards(r.off-pred_orin_edge_start+k,
                                                  edge_headers,
                                                  pred_edge_shard_size);
            if (edge_addr[ptr].val==end)
            {
                index_list_mirror[index] = 1;
                break;
            }
        }
   }


}

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
                    cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_get_edge_list_k2c << < dimGrid, dimBlock, 0, stream_id  >> > (
                    slot_id_list,
                    (vertex_t*)d_vertex_addr,
                    index_list,
                    index_list_mirror,
                    ptr_list,
                    query_size,
                    (edge_t*)edge_addr,
                    end,
                    pred_orin_edge_start,
                    edge_headers,
                    pred_edge_shard_size);

}

__global__
void d_update_result_table_i2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  edge_t *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;

    int edge_num = 0;
    edge_num = index_list[0];

    if(index<edge_num) {
            uint64_t ptr = map_location_on_shards(ptr_list[0]+index,
                                                  edge_headers,
                                                  pred_edge_shard_size);
            //printf("ptr:%d\n",(&(edge_addr[ptr])+index)->val);
            updated_result_table[index] = edge_addr[ptr].val;
    }

}

int update_result_table_i2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  void *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size,
                                  cudaStream_t stream_id)

{
    int table_size = 0;//index_list[query_size-1];
    CUDA_SAFE_CALL( cudaMemcpyAsync(&table_size,
               index_list,
               sizeof(int),
               cudaMemcpyDeviceToHost, stream_id) );

    int gridsize = (int) (ceil((double)table_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_update_result_table_i2u<< < dimGrid, dimBlock, 0, stream_id  >> > (result_table,
         updated_result_table,
         index_list,
         ptr_list,
         (edge_t*)edge_addr,
         edge_headers,
         pred_edge_shard_size
         );

    CUDA_SAFE_CALL( cudaStreamSynchronize(stream_id) );
    return table_size;

}

__global__
void d_update_result_table_k2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  edge_t *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size,
                                  int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;

    //int index = full_index/200/(column_num+1);
    if(index<query_size) {
        //extern __shared__ int result_matrix[];

        int edge_num = 0,start=0;
        if(index==0) {
            edge_num = index_list[index];
            start = 0;
        }
        else {
            edge_num = index_list[index] - index_list[index - 1];
            start = (column_num+1)*index_list[index - 1];
        }

        int buff[20];
        for(int c=0;c<column_num;c++){
            buff[c] = result_table[column_num*index+c];
        }

        for(int k=0;k<edge_num;k++){
            for(int c=0;c<column_num;c++){
                updated_result_table[start+k*(column_num+1)+c] = buff[c];//result_table[column_num*index+c];
            }
            uint64_t ptr = map_location_on_shards(ptr_list[index]+k,
                                                  edge_headers,
                                                  pred_edge_shard_size);

            updated_result_table[start+k*(column_num+1)+column_num] = edge_addr[ptr].val;
        }
    }

}

int update_result_table_k2u(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  void *edge_addr,
                                  uint64_t* edge_headers,
                                  uint64_t pred_edge_shard_size,
                                  int query_size,
                                  cudaStream_t stream_id)

{
    int table_size = 0;//index_list[query_size-1];
    CUDA_SAFE_CALL( cudaMemcpyAsync(&table_size,
               index_list+query_size-1,
               sizeof(int),
               cudaMemcpyDeviceToHost, stream_id) );

    //query_size = query_size*200*(column_num+1);
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_update_result_table_k2u<< < dimGrid, dimBlock, 0, stream_id  >> > (result_table,
         updated_result_table,
         index_list,
         ptr_list,
         column_num,
         (edge_t*)edge_addr,
         edge_headers,
         pred_edge_shard_size,
         query_size);

    CUDA_SAFE_CALL( cudaStreamSynchronize(stream_id) );
    return table_size*(column_num+1);

}


__global__
void d_update_result_table_k2k(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  edge_t *edge_addr,
                                  int end,
                                  int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;

    if(index<query_size) {
        int edge_num = 0,start=0;
        if(index==0) {
            edge_num = index_list[index];
            start = 0;
        }
        else {
            edge_num = index_list[index] - index_list[index - 1];
            start = column_num*index_list[index - 1];
        }
        int buff[20];
        for(int c=0;c<column_num;c++){
            buff[c] = result_table[column_num*index+c];
        }
        for(int k=0;k<edge_num;k++){
            for(int c=0;c<column_num;c++){
                updated_result_table[start+c] = buff[c];//result_table[column_num*index+c];
            }
        }
    }

}

int update_result_table_k2k(int *result_table,
                                  int *updated_result_table,
                                  int *index_list,
                                  uint64_t *ptr_list,
                                  int column_num,
                                  void *edge_addr,
                                  int end,
                                  int query_size,
                                  cudaStream_t stream_id)

{
    int table_size = 0;//index_list[query_size-1];
    CUDA_SAFE_CALL( cudaMemcpyAsync(&table_size,
               index_list+query_size-1,
               sizeof(int),
               cudaMemcpyDeviceToHost, stream_id) );

    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_update_result_table_k2k<< < dimGrid, dimBlock, 0, stream_id  >> > (result_table,
         updated_result_table,
         index_list,
         ptr_list,
         column_num,
         (edge_t*)edge_addr,
         end,
         query_size);

    CUDA_SAFE_CALL( cudaStreamSynchronize(stream_id) );
    return table_size*column_num;

}


void calc_prefix_sum(int* d_out_arr,
                     int* d_in_arr,
                     int query_size,
                     cudaStream_t stream_id)
{
    thrust::device_ptr<int> d_in_arr_ptr(d_in_arr);
    thrust::device_ptr<int> d_out_arr_ptr(d_out_arr);
    thrust::inclusive_scan(thrust::cuda::par.on(stream_id), d_in_arr_ptr, d_in_arr_ptr + query_size, d_out_arr_ptr);

}


// Siyuan: 计算history中每条record(每行)的目的地server id
__global__
void d_hash_dispatched_server_id(int *result_table,
                                  int *server_id_list,
                                  int start,
                                  int col_num,
                                  int num_sub_request,
                                  int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<query_size) {
        // Siyuan: index指定是第几行
        server_id_list[index] =  result_table[index * col_num + (-start - 1)] % num_sub_request;
    }

}

void hash_dispatched_server_id(int *result_table,
                                  int *server_id_list,
                                  int start,
                                  int col_num,
                                  int num_sub_request,
                                  int query_size,
                                  cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_hash_dispatched_server_id << < dimGrid, dimBlock, 0, stream_id >> > (result_table,
                                  server_id_list,
                                  start,
                                  col_num,
                                  num_sub_request,
                                  query_size);
}

__global__
void d_history_dispatch(int *result_table,
                        int* position_list,
                        int* server_id_list,
                        int* server_sum_list,
                        int start,
                        int col_num,
                        int num_sub_request,
                        int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if(index<query_size) {
        int server_id =server_id_list[index];
        position_list[index] = atomicAdd(&server_sum_list[server_id],1);
    }

}

void history_dispatch(int *result_table,
                        int* position_list,
                        int* server_id_list,
                        int* server_sum_list,
                        int start,
                        int col_num,
                        int num_sub_request,
                        int query_size,
                        cudaStream_t stream_id)
{

    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_history_dispatch << < dimGrid, dimBlock, 0, stream_id >> >(result_table,
                                                   position_list,
                                                   server_id_list,
                                                   server_sum_list,
                                                   start,
                                                   col_num,
                                                   num_sub_request,
                                                   query_size);
}



void calc_dispatched_position(int *d_result_table,
                              int *d_position_list,
                              int *d_server_id_list,
                              int *d_server_sum_list,
                              int *gpu_sub_table_size_list,
                              int start,
                              int column_num,
                              int num_sub_request,
                              int query_size,
                              cudaStream_t stream_id)
{
    // Siyuan: 计算每条record将要被发送到的server id
    hash_dispatched_server_id(d_result_table,
                                  d_server_id_list,
                                  start,
                                  column_num,
                                  num_sub_request,
                                  query_size,
                                  stream_id);

    // Siyuan: 此处是把parent history table切分成child history table
    history_dispatch(d_result_table,
                     d_position_list,
                         d_server_id_list,
                         d_server_sum_list,
                         start,
                         column_num,
                         num_sub_request,
                         query_size,
                         stream_id);

    // Siyuan: gpu_sub_table_size_list中存的是每个sub table的
    CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_sub_table_size_list,
                                  d_server_sum_list,
                                  sizeof(int) * num_sub_request,
                                  cudaMemcpyDeviceToHost,
                                  stream_id));
    cudaStreamSynchronize(stream_id);

    // Siyuan: 对d_server_sum_list计算exclusive的前置和
    thrust::device_ptr<int> d_server_sum_list_ptr(d_server_sum_list);
    thrust::exclusive_scan(thrust::cuda::par.on(stream_id), d_server_sum_list_ptr, d_server_sum_list_ptr + num_sub_request, d_server_sum_list_ptr);
    // 函数返回之后d_server_sum_list中就是[0,5,12]这样的前值和
}


// Siyuan: updated_result_table是一个device上的大buffer，
// parent history table通过不同的偏移量把sub query table映射到这个buffer中
__global__
void d_update_result_table_sub(int *result_table,
                  int *updated_result_table,
                  int *d_position_list,
                  int *server_id_list,
                  int *sub_table_hdr_list,
                  int column_num,
                  int num_sub_request,
                  int query_size)
{
    int index = blockIdx.x * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    if (index < query_size) {
        int dst_sid = server_id_list[index];
        int mapped_index = sub_table_hdr_list[dst_sid] + d_position_list[index];
        for (int c = 0; c < column_num; c++) {
            updated_result_table[column_num * mapped_index + c] = result_table[column_num * index + c];
        }
    }
}


void update_result_table_sub(int *result_table,
                                  int *updated_result_table,
                                  int *d_position_list,
                                  int *server_id_list,
                                  int *sub_table_hdr_list,
                                  int column_num,
                                  int num_sub_request,
                                  int query_size,
                                  cudaStream_t stream_id)
{
    int gridsize = (int) (ceil((double)query_size / (blocksize * blocksize)));
    dim3 dimBlock = dim3(blocksize, blocksize, 1);
    dim3 dimGrid= dim3(gridsize, 1, 1);

    d_update_result_table_sub<< < dimGrid, dimBlock, 0, stream_id >> > (result_table,
         updated_result_table,
         d_position_list,
         server_id_list,
         sub_table_hdr_list,
         column_num,
         num_sub_request,
         query_size);
}


