#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>
#include <cstdio>

#include "mem.hpp"
#include "string_server.hpp"
#include "dgraph.hpp"
#include "engine.hpp"
#include "gpu_engine.hpp"
#include "proxy.hpp"
#include "console.hpp"
#include "monitor.hpp"
#include "rdma_resource.hpp"
#include "adaptor.hpp"

#include "unit.hpp"
#include "timer.hpp"

#include "data_statistic.hpp"
#include "gpu_mem.hpp"


#define NUM_PACKETS 5
#define MB (1024 * 1024)


struct packet {
    char *ptr;
    char *devPtr;
    uint32_t length;
};

packet pks[NUM_PACKETS];

void fillBuf(char *ptr, uint32_t len) {
    char a = 'a';
    for (int i = 0; i < len; ++i) {
        ptr[i] = a + rand() % 26;
    }
}

void set_config() {
    global_memstore_size_gb = 1;
    global_num_servers = 2;
    global_num_gpu_engines = global_num_threads = 2;
    global_num_engines = 0;
    global_num_proxies = 0;
}

/**
 * 写一个benchmark
 * 1) 生成10个packet，放到GPU上
 * 2）从GPU上把数据拷贝到CPU上
 * 3）local CPU --> remote CPU
 * 4) remote CPU --> remote GPU
 */
int main(int argc, char *argv[])
{
    char *devPtr;
    uint32_t size = 1;
    uint64_t t1, t2, t3, t4;
    int sid;

    if (argc < 2) {
        printf("Usage: %s sid\n", argv[0]);
        exit(1);
    }

    sid = atoi(argv[1]);
    set_config();

    cout << "I am server " << sid << endl;

	// allocate memory
	Mem *mem = new Mem(global_num_servers, global_num_threads);
    GPUMem *gmem = new GPUMem(global_num_servers, global_num_gpu_engines);
	cout << "INFO#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << endl;

	// init RDMA devices and connections
	RDMA_init(global_num_servers, global_num_threads, sid,
            mem->memory(), mem->memory_size(), gmem->memory(), gmem->memory_size(), "mpd.hosts");

	// init data communication
	RDMA_Adaptor *rdma_adaptor = nullptr;
	rdma_adaptor = new RDMA_Adaptor(sid, mem, global_num_servers, global_num_threads);

    // data prepare
    for (int i = 0; i < NUM_PACKETS; ++i) {
        pks[i].length = (size << i) * MB;
        pks[i].ptr = malloc(pks[i].length);
        fillBuf(pks[i].ptr, pks[i].length);
    }

    // put data onto GPU
    for (int i = 0; i < NUM_PACKETS; ++i) {
        GPU_ASSERT( cudaMalloc(&(pks[i].devPtr), pks[i].length) );
        GPU_ASSERT( cudaMemcpy(pks[i].devPtr, pks[i].ptr, pks[i].length, cudaMemcpyHostToDevice) );
        memset(pks[i].ptr, 0, pks[i].length);

    }

    for (int i = 0; i < NUM_PACKETS; ++i) {
        // #1 copy data from GPU to CPU
        t1 = timer::get_usec();
        GPU_ASSERT( cudaMemcpy(pks[i].ptr, pks[i].devPtr, pks[i].length, cudaMemcpyDeviceToHost) );
        printf("[%d] Device to host: %luus\n", i, timer::get_usec() - t1);

        t2 = timer::get_usec();
        string msg(pks[i].ptr, pks[i].length);
        t3 = timer::get_usec();
        // #2 send to receiver
        rdma_adaptor->send(0, 1, 0, msg); // (tid, dst_sid, dst_tid, buf)
        t4 = timer::get_usec();
        printf("[%d] RDMA sent: %luus\n", i, t4 - t3);
    }



    cout << "done" << endl;

    return 0;
}


