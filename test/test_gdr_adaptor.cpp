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


struct data_payload {
    char str[30];
};


class Mock_Engine {

    Adaptor *adaptor;
    int sid;

public:
    Mock_Engine(int sid, Adaptor *adaptor) {
        this->sid = sid;
        /* this->id = id; */
        this->adaptor = adaptor;
    }

    void run() {
        request_or_reply to_send;
        request_or_reply recved;
        char *devPtr;


        if (sid == 0) {  // Bob
            printf("I am Bob\n");
            GPU_ASSERT( cudaMalloc(&devPtr, 128) );

            char str[] = "hello world from Bob";
            data_payload payload;
            memcpy(payload.str, str, sizeof(str));

            GPU_ASSERT( cudaMemcpy(devPtr, &payload, sizeof(data_payload), cudaMemcpyHostToDevice) );
            to_send.gpu_history_table_ptr = devPtr;
            to_send.gpu_history_table_size = sizeof(data_payload);
            to_send.cmd_chains.push_back(8);
            to_send.cmd_chains.push_back(9);
            to_send.cmd_chains.push_back(10);

            adaptor->gpu_send(1, to_send);

        } else {    // Alice
            printf("I am Alice\n");
            GPU_ASSERT( cudaMalloc(&devPtr, 128) );
            recved = adaptor->gpu_recv(devPtr, 128);
            char buf[128];



            printf("Alice recevied request:\n");
            printf("history_table: %p\n", recved.gpu_history_table_ptr);
            printf("history_table_sz: %lu\n", recved.gpu_history_table_size);
            GPU_ASSERT( cudaMemcpy(buf, recved.gpu_history_table_ptr, recved.gpu_history_table_size, cudaMemcpyDeviceToHost) );
            printf("history content: %s\n", buf);

            if (recved.cmd_chains.empty() || recved.cmd_chains.size() < 3) {
                printf("cmd_chains not received corretly!\n");
            } else {
                printf("cmd_chains: %d %d %d\n", recved.cmd_chains[0], recved.cmd_chains[1],
                        recved.cmd_chains[2]);
            }
        }
    }

};

void set_config() {
    global_memstore_size_gb = 1;
    global_num_servers = 2;
    global_num_gpu_engines = 2;
    global_num_engines = 4;
    global_num_proxies = 4;

    global_num_threads = global_num_engines + global_num_proxies + global_num_gpu_engines;
}

void *thread_func(void *arg) {
    Mock_Engine *engine = (Mock_Engine *)arg;
    engine->run();

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

    // 可以跑通的config
    // global_num_servers 2
    // global_num_threads 2
    // global_num_engines 0
    // global_num_proxies 0
    cout << "global_num_servers: " << global_num_servers << endl;
    cout << "global_num_threads: " << global_num_threads << endl;
    cout << "global_num_engines: " << global_num_engines << endl;
    cout << "global_num_proxies: " << global_num_proxies << endl;

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
        gdr_adaptor = new GDR_Adaptor(sid, gpu_mem, global_num_servers, global_num_gpu_engines);
    }

	pthread_t *threads  = new pthread_t[global_num_threads];


    int tid = 0 + global_num_engines + global_num_proxies;
    Adaptor *adaptor = new Adaptor(tid, NULL, rdma_adaptor, gdr_adaptor);

    Mock_Engine *engine = new Mock_Engine(sid, adaptor);
    pthread_create(&(threads[tid]), NULL, thread_func, (void *)engine);

    /* for (int tid = 0; tid < global_num_gpu_engines; tid++) { */
        int rc = pthread_join(threads[tid], NULL);
        if (rc) {
            printf("pthread_join error\n");
            exit(1);
        }
    /* } */

    return 0;
}

