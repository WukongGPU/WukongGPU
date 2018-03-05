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
 */

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <tbb/concurrent_unordered_map.h>
#include <iostream>
#include <map>
#include <string.h>

#include "mem.hpp"
#include "string_server.hpp"
#include "dgraph.hpp"
#include "engine.hpp"
#include "proxy.hpp"
#include "console.hpp"
#include "monitor.hpp"
#include "rdma_transport.hpp"
#include "adaptor.hpp"

#include "unit.hpp"

#include "data_statistic.hpp"

#include "gpu_mem.hpp"
#include "gdr_transport.hpp"
#include "gpu_engine.hpp"
#include "rdf_meta.hpp"
#include "gpu_stream.hpp"
#include "rcache.hpp"
#include "agent_adaptor.hpp"
#include "taskq_meta.hpp"

using namespace std;

/*
 * The processor architecture of our cluster (meepo0-4)
 *
 * $numactl --hardware
 * available: 2 nodes (0-1)
 * node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46
 * node 0 size: 128836 MB
 * node 0 free: 127168 MB
 * node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47
 * node 1 size: 129010 MB
 * node 1 free: 127922 MB
 * node distances:
 * node   0   1
 *   0:  10  21
 *   1:  21  10
 *
 * TODO:
 * co-locate proxy and engine threads to the same socket,
 * and bind them to the same socket. For example, 2 proxy thread with
 * 8 engine threads for each 10-core processor.
 *
 */
int cores[] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
};

// for multiple instances
int numa_nodes[][12] = {
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22},
    {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
};

bool monitor_enable = false;
int monitor_port = 5450;

void __pin_to_core(size_t core)
{
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(core, &mask);
	int result = sched_setaffinity(0, sizeof(mask), &mask);
}

void pin_to_core(int sid, int tid)
{
    if (global_multi_instance) {
        if (sid % NUM_INSTANCES == 0) {
            __pin_to_core(numa_nodes[0][tid]);
        } else {
            __pin_to_core(numa_nodes[1][tid]);
        }

    } else {
        __pin_to_core(cores[tid]);
    }
}

void *engine_thread(void *arg)
{
	Engine *engine = (Engine *)arg;
    pin_to_core(engine->sid, engine->tid);
    engine->run();
}

void *gpu_engine_thread(void *arg)
{
	GPU_Engine *engine = (GPU_Engine *)arg;
    GPU_ASSERT( cudaSetDevice(engine->devid) );
    pin_to_core(engine->sid, engine->tid);
    engine->run();
}

void *proxy_thread(void *arg)
{
	Proxy *proxy = (Proxy *)arg;
	pin_to_core(proxy->sid, proxy->tid);
	if (!monitor_enable)
		// Run the Wukong's testbed console (by default)
		run_console(proxy);
	else
		// Run monitor thread for clients
		run_monitor(proxy, monitor_port);
}

static void
usage(char *fn)
{
	cout << "usage: << fn <<  <config_fname> <host_fname> [options]" << endl;
	cout << "options:" << endl;
	cout << "  -c: enable connected client" << endl;
	cout << "  -p port_num : the port number of connected client (default: 5450)" << endl;
}

static void
send_pred_metas(int sid, TCP_Transport *tcp, DGraph &dgraph)
{
    std::stringstream ss;
    std::string str;
    boost::archive::binary_oarchive oa(ss);
    Pred_Metas_Msg msg(dgraph.gstore.get_pred_metas());

    msg.sid = sid;

    oa << msg;

    // send pred_metas to other servers
    for (int i = 0; i < global_num_servers; ++i) {
        if (i == sid)
            continue;
        tcp->send(i, 0, ss.str());
        cout << "INFO#"<< sid << " send pred metas to server " << i << endl;
    }

}

static void
recv_pred_metas(int sid, TCP_Transport *tcp, tbb::concurrent_unordered_map<int, vector<pred_meta_t> > &global_pred_metas)
{
    std::string str;

    // receive #global_num_servers - 1 messages
    for (int i = 0; i < global_num_servers; ++i) {
        if (i == sid)
            continue;
        std::stringstream ss;
        str = tcp->recv(0);
        ss << str;
        boost::archive::binary_iarchive ia(ss);
        Pred_Metas_Msg msg;
        ia >> msg;

        global_pred_metas.insert(make_pair(msg.sid, msg.data));
        cout << "INFO#" << sid << " recv pred metas from server " << msg.sid << endl;
    }

}

int
main(int argc, char *argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	int sid = world.rank(); // server ID
    int devid = 0;


	if (argc < 3) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	cfg_fname = std::string(argv[1]);
	host_fname = std::string(argv[2]);

	int c;
	while ((c = getopt(argc - 2, argv + 2, "cp:")) != -1) {
		switch (c) {
		case 'c':
			monitor_enable = true;
			break;
		case 'p':
			monitor_port = atoi(optarg);
			break;
		default :
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	// load global configuration setting
	load_config(world.size());

    extern struct config_t rdma_config;
    if (global_multi_instance) {
        cout << "INFO#" << sid << " running in multiple instances mode" << endl;
        char *ibdevs[] = {"mlx4_0", "mlx4_1"};
        devid = sid % NUM_INSTANCES;
        strncpy(rdma_config.dev_name, ibdevs[devid], strlen(ibdevs[devid]) + 1);
    }
    cout << "INFO#" << sid << ": using GPU" << devid << " and NIC " << rdma_config.dev_name << endl;


	// allocate memory
	Mem *mem = new Mem(global_num_servers, global_num_threads);
	cout << "INFO#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << endl;
    GPUMem *gpu_mem = new GPUMem(devid, global_num_servers, global_num_gpu_engines);
	cout << "INFO#" << sid << ": allocate " << B2GiB(gpu_mem->memory_size()) << "GB GPU memory" << endl;

#ifdef CUDA_DEBUG
    {
        int i = 0;
        char host[256];
        printf("PID %d on node %d is ready for attach\n", getpid(), sid);
        fflush(stdout);
        while (0 == i) {
         sleep(5);
        }
    }
#endif



	// init RDMA devices and connections
#ifdef HAS_RDMA
	RDMA_init(global_num_servers, global_num_threads, sid,
            mem->memory(), mem->memory_size(), gpu_mem->memory(), gpu_mem->memory_size(), host_fname);
#endif

	// init data communication
	RDMA_Transport *rdma_transport = NULL;
    GDR_Transport *gdr_transport = nullptr;
	if (RDMA::get_rdma().has_rdma()) {
		rdma_transport = new RDMA_Transport(sid, mem, global_num_servers, global_num_threads);
        gdr_transport = new GDR_Transport(sid, gpu_mem, mem, global_num_servers, global_num_gpu_engines);
    }

	TCP_Transport *tcp_transport = new TCP_Transport(sid, host_fname, global_num_threads, global_data_port_base);

	// load string server (read-only, shared by all proxies)
	String_Server str_server(global_input_folder);

	// load RDF graph (shared by all engines)
	DGraph dgraph(sid, mem, global_input_folder);

    StreamPool streamPool(global_num_cuda_streams);

	// load RDF cache (shared by all engines)
	RCache rcache(devid, &dgraph, &streamPool, sid);

    tbb::concurrent_unordered_map<int, vector<pred_meta_t> > global_pred_metas;
    // std::map<int, std::vector<pred_meta_t>> global_pred_metas;

    // synchronize predicate metadatas
    send_pred_metas(sid, tcp_transport, dgraph);
    recv_pred_metas(sid, tcp_transport, global_pred_metas);
    dgraph.gstore.set_global_pred_metas(global_pred_metas);

    TaskQ_Meta::init(global_num_servers, global_num_threads);
    GPU::instance().init(&dgraph.gstore, &rcache, rcache.shardmanager);


  // prepare data for planner
  // data_statistic stat(tcp_adaptor, &world);
  // if (global_enable_planner) {
      // dgraph.gstore.generate_statistic(stat);
      // stat.gather_data();
  // }

	// init control communicaiton
    con_adaptor = new TCP_Transport(sid, host_fname, global_num_proxies, global_ctrl_port_base);

	// launch proxy and engine threads
	assert(global_num_threads == global_num_proxies + global_num_engines + global_num_gpu_engines);
	pthread_t *threads  = new pthread_t[global_num_threads];
	for (int tid = 0; tid < global_num_proxies + global_num_engines; tid++) {
    		Adaptor *adaptor = new Adaptor(tid, tcp_transport, rdma_transport);
		if (tid < global_num_proxies) {
			Proxy *proxy = new Proxy(sid, tid, &str_server, adaptor, nullptr, mem);
			pthread_create(&(threads[tid]), NULL, proxy_thread, (void *)proxy);
			proxies.push_back(proxy);
		} else {
			Engine *engine = new Engine(sid, tid, &dgraph, adaptor);
			pthread_create(&(threads[tid]), NULL, engine_thread, (void *)engine);
			engines.push_back(engine);
		}
	}

    // launch gpu engine thread
    for (int tid = global_num_proxies + global_num_engines; tid < global_num_threads; tid++) {
            Agent_Adaptor *adaptor = new Agent_Adaptor(tid, tcp_transport, rdma_transport, gdr_transport);
			GPU_Engine *gpu_engine = new GPU_Engine(devid, sid, tid, &rcache, adaptor);
			pthread_create(&(threads[tid]), NULL, gpu_engine_thread, (void *)gpu_engine);
            gpu_engines.push_back(gpu_engine);
    }

	// wait to all threads termination
	for (size_t t = 0; t < global_num_threads; t++) {
		int rc = pthread_join(threads[t], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

	/// TODO: exit gracefully (properly call MPI_Init() and MPI_Finalize(), delete all objects)
	return 0;
}
