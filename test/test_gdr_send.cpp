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

#include "data_statistic.hpp"
#include "gpu_mem.hpp"
#include "gdr_transport.hpp"


#define NUM_PACKETS 3
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
    /* global_memstore_size_gb = 1;
     * global_num_servers = 2;
     * global_num_gpu_engines = global_num_threads = 2;
     * global_num_engines = 0;
     * global_num_proxies = 0; */


    global_memstore_size_gb = 1;
    global_num_servers = 2;
    global_num_gpu_engines = 1;
    global_num_engines = 1;
    global_num_proxies = 0;

    global_num_threads = 2;
}

class gpu_engine {
public:
    RDMA_Transport *rdma_adaptor;
    GDR_Transport * gdr_adaptor;
    int sid;
    int tid;

    gpu_engine(RDMA_Transport *rdma_adaptor, GDR_Transport * gdr_adaptor, int sid)
        : rdma_adaptor(rdma_adaptor), gdr_adaptor(gdr_adaptor), sid(sid)
    {  }

    void run() {
        char *devPtr;
        uint64_t t1, t2;
        uint64_t size = 1;


        printf("gpu_engine is running\n");
        // data prepare
        for (int i = 0; i < NUM_PACKETS; ++i) {
            pks[i].length = (size << i) * MB;
            pks[i].ptr = malloc(pks[i].length);
            fillBuf(pks[i].ptr, pks[i].length);
        }

        GPU_ASSERT( cudaMalloc(&devPtr, 100 * MB) );
        // allocate device memory
        pks[0].devPtr = devPtr;
        for (int i = 1; i < NUM_PACKETS - 1; ++i) {
            pks[i].devPtr = pks[i-1].devPtr + (size << (i-1)) * MB;
        }

        // put data onto GPU
        for (int i = 0; i < NUM_PACKETS; ++i) {
            GPU_ASSERT( cudaMalloc(&(pks[i].devPtr), pks[i].length) );
            GPU_ASSERT( cudaMemcpy(pks[i].devPtr, pks[i].ptr, pks[i].length, cudaMemcpyHostToDevice) );
            memset(pks[i].ptr, 0, pks[i].length);
        }

        for (int i = 0; i < NUM_PACKETS; ++i) {
            // GDR send to receiver
            t1 = timer::get_usec();
            assert(sid != 1);
            gdr_adaptor->send(0, 1, 1, pks[i].devPtr, pks[i].length, rdma_mem_t(GPU_DRAM, CPU_DRAM));
            t2 = timer::get_usec();
            printf("[%d] GDR sent: %lu bytes, %luus\n", i, pks[i].length, t2 - t1);
        }


        cout << "Sender done" << endl;
    }



};

class engine {
public:
    RDMA_Transport *rdma_adaptor;
    GDR_Transport * gdr_adaptor;
    int tid;

    engine(RDMA_Transport *rdma_adaptor, GDR_Transport * gdr_adaptor)
        : rdma_adaptor(rdma_adaptor), gdr_adaptor(gdr_adaptor)
    {  }

    void run() {
        printf("engine is running\n");
        while (1) {
            timer::cpu_relax(200);
        }
    }

};

void *engine_thread(void *arg)
{
    engine *e = (engine *)arg;
    e->run();
}

void *gpu_engine_thread(void *arg)
{
    gpu_engine *e = (gpu_engine *)arg;
    GPU_ASSERT( cudaSetDevice(0) );
    e->run();
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

    cout << "I am server " << sid << endl;
    set_config();

    // global_num_servers 2
    // global_num_threads 2
    // global_num_engines 0
    // global_num_proxies 0

	// allocate memory
	Mem *mem = new Mem(global_num_servers, global_num_threads);
	cout << "INFO#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << endl;
    GPUMem *gpu_mem = new GPUMem(0, global_num_servers, global_num_gpu_engines);
	cout << "INFO#" << sid << ": allocate " << B2GiB(gpu_mem->memory_size()) << "GB GPU memory" << endl;


	// init RDMA devices and connections
	RDMA_init(global_num_servers, global_num_threads, sid,
            mem->memory(), mem->memory_size(), gpu_mem->memory(), gpu_mem->memory_size(), "mpd.hosts");


	// init data communication
	RDMA_Transport *rdma_adaptor = NULL;
    GDR_Transport *gdr_adaptor = nullptr;
	if (RDMA::get_rdma().has_rdma()) {
		rdma_adaptor = new RDMA_Transport(sid, mem, global_num_servers, global_num_threads);
        /* GDR_Transport最后一个参数传global_num_threads是为了创建与
         * RDMA_Transport中一样数量的rmetas (num_servers x num_threads)
         */
        gdr_adaptor = new GDR_Transport(sid, gpu_mem, mem, global_num_servers, global_num_threads);
    }


	pthread_t *threads  = new pthread_t[global_num_threads];
	for (int tid = 0; tid < global_num_engines + global_num_gpu_engines; tid++) {
		if (tid < global_num_engines) {
			engine *e = new engine(rdma_adaptor, gdr_adaptor);
            e->tid = tid;
			pthread_create(&(threads[tid]), NULL, engine_thread, (void *)e);
		} else {
            gpu_engine *ge = new gpu_engine(rdma_adaptor, gdr_adaptor, sid);
            ge->tid = tid;
			pthread_create(&(threads[tid]), NULL, gpu_engine_thread, (void *)ge);
		}
	}

	// wait to all threads termination
	for (size_t t = 0; t < global_num_threads; t++) {
		int rc = pthread_join(threads[t], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

    cout << "Sender bye" << endl;

    return 0;
}

