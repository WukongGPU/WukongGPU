#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>
#include <cstdio>

#include "config.hpp"
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

int main(int argc, char *argv[])
{
	/* boost::mpi::environment env(argc, argv);
	 * boost::mpi::communicator world; */
    if (argc < 2) {
        printf("%s sid\n", argv[0]);
        exit(1);
    }

	int sid = atoi(argv[1]); // server ID
    char *devPtr;
    uint64_t t1, t2, t3, t4;

    cout << "I am server " << sid << endl;
    set_config();

    // global_num_servers 2
    // global_num_threads 2
    // global_num_engines 0
    // global_num_proxies 0

	// allocate memory
	Mem *mem = new Mem(global_num_servers, global_num_threads);
	cout << "INFO#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << endl;
    GPUMem *gpu_mem = new GPUMem(global_num_servers, global_num_gpu_engines);
	cout << "INFO#" << sid << ": allocate " << B2GiB(gpu_mem->memory_size()) << "GB GPU memory" << endl;


	// init RDMA devices and connections
	RDMA_init(global_num_servers, global_num_threads, sid,
            mem->memory(), mem->memory_size(), gpu_mem->memory(), gpu_mem->memory_size(), "mpd.hosts");


	// init data communication
	RDMA_Adaptor *rdma_adaptor = NULL;
    GDR_Adaptor *gdr_adaptor = nullptr;
	if (RDMA::get_rdma().has_rdma()) {
		rdma_adaptor = new RDMA_Adaptor(sid, mem, global_num_servers, global_num_threads);
        gdr_adaptor = new GDR_Adaptor(sid, gpu_mem, global_num_servers, global_num_threads);
    }


    GPU_ASSERT( cudaMalloc(&devPtr, 100 * MB) );

    // GDR send to receiver
    uint64_t data_sz;
    for (int i = 0; i < NUM_PACKETS; ++i) {
        t1 = timer::get_usec();
        data_sz = gdr_adaptor->recv(0, 0, devPtr, 100 * MB);
        t2 = timer::get_usec();
        printf("[%d] Received: %lu bytes, %luus\n", i, data_sz, t2 - t1);
        /* printf("[%d] Time elapsed: %lu us\n", i, timer::get_usec() - t1); */
    }

    cout << "Receiver done" << endl;

    return 0;
}
