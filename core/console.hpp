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

#include <iostream>
#include <string>
#include <set>
#include <boost/unordered_map.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "config.hpp"
#include "proxy.hpp"
#include "logger.hpp"

using namespace std;

// communicate between proxy threads
TCP_Transport *con_adaptor;

template<typename T>
static void console_send(int sid, int tid, T &r) {
	std::stringstream ss;
	boost::archive::binary_oarchive oa(ss);
	oa << r;
	con_adaptor->send(sid, tid, ss.str());
}

template<typename T>
static T console_recv(int tid) {
	std::string str;
	str = con_adaptor->recv(tid);

	std::stringstream ss;
	ss << str;

	boost::archive::binary_iarchive ia(ss);
	T r;
	ia >> r;
	return r;
}

static void console_barrier(int tid)
{
	static int _curr = 0;
	static __thread int _next = 1;

	// inter-server barrier
	if (tid == 0)
		MPI_Barrier(MPI_COMM_WORLD);

	// intra-server barrier
	__sync_fetch_and_add(&_curr, 1);
	while (_curr < _next)
		usleep(1); // wait
	_next += global_num_proxies; // next barrier
}

void print_help(void)
{
	cout << "These are common Wukong commands: " << endl;
	cout << "    help           Display help infomation" << endl;
	cout << "    quit           Quit from console" << endl;
    cout << "    stat-gdr       Print statistics of GPUDirect RDMA" << endl;
	cout << "    reload-config  Reload config file" << endl;
	cout << "    show-config    Show current config" << endl;
    cout << "    run-script     Run commands in 'wukong.script' file" << endl;
	cout << "    sparql         Run SPARQL queries" << endl;
	cout << "        -f <file>   a single query from the <file>" << endl;
	cout << "        -n <num>    run a single query <num> times" << endl;
	cout << "        -b <file> [<args>]  run queries configured by <file> (batch-mode)" << endl;
	cout << "           -d <sec>            eval <sec> seconds" << endl;
	cout << "           -w <sec>            warmup <sec> seconds" << endl;
	cout << "           -i <msec>           sleep <msec> before sending next query" << endl;
	cout << "           -p <num>            send <num> queries in parallel" << endl;

	cout << "        -s <string> a single query from input string" << endl;
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define IS_MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
#define PRINT_ID(_p) (cout << "[" << (_p)->sid << "-" << (_p)->tid << "]$ ")


/**
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void run_console(Proxy *proxy)
{
	console_barrier(proxy->tid);
	if (IS_MASTER(proxy))
		cout << endl
		     << "Input \'help\'' command to get more information"
		     << endl
		     << endl;

    // ifstream ifs("lubm_input");
    // if (!ifs) {
        // std::cout << "Open lubm_input fail" << endl;
        // assert(false);
    // }
    ifstream file_script;
    // Logger master_logger;
    // logger每跑完一次就填充一下这些Figure所需的数据
    Fig_Thpt_Latency fig_thpt_latency;

	while (true) {
		console_barrier(proxy->tid);
next:
		string cmd;
		if (IS_MASTER(proxy)) {

			cout << "> ";
            // read command from file first
            if (file_script.is_open() && !file_script.eof()) {
                sleep(3);
                getline(file_script, cmd);
                cout << cmd << endl;;
            } else {
                if (file_script.is_open())
                    file_script.close();
                std::getline(std::cin, cmd);
            }

			// trim input
			size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
			if (pos == string::npos) goto next;
			cmd.erase(0, pos);

			pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
			cmd.erase(pos + 1, cmd.length() - (pos + 1));

			// only process on the master console
			if (cmd == "help") {
				print_help();
				goto next;
			} else if (cmd == "show-config") {
				show_config();
				goto next;
			} else if (cmd == "run-script") {
                file_script.open("wukong.script");
                goto next;
            }

			// send commands to all proxy threads
			for (int i = 0; i < global_num_servers; i++) {
				for (int j = 0; j < global_num_proxies; j++) {
					if (i == 0 && j == 0)
						continue ;
					console_send<string>(i, j, cmd);
				}
			}
		} else {
			// recieve commands
			cmd = console_recv<string>(proxy->tid);
		}

		// process on all consoles
        if (cmd == "stat-gdr") {
            if (IS_MASTER(proxy)) {
                printf("Proxy[%d:%d] GPUDirect RDMA: data_sz=%d B, #ops=%d\n",
                        proxy->sid, proxy->tid, global_stat.gdr_data_sz, global_stat.gdr_times);
                global_stat.reset_gdr_stats();
            }
            goto next;

        } if (cmd == "quit" || cmd == "q") {
			if (proxy->tid == 0)
				exit(0); // each server exits once by the 1st proxy thread
		} else if (cmd == "reload-config") {
			if (proxy->tid == 0)
				reload_config(); // each server reload config file once by the 1st proxy
		} else {
			std::stringstream cmd_ss(cmd);
			std::string token;

			// get keyword of command
			cmd_ss >> token;

			// handle SPARQL queries
			if (token == "sparql") {
				string fname, bfname, query;
				int cnt = 1;
				int duration = 10, warmup = 5, parallel_factor = 20, send_interval = 0;
				bool f_enable = false, b_enable = false, q_enable = false;

				// parse parameters
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
					} else if (token == "-n") {
						cmd_ss >> cnt;
					} else if (token == "-b") {
						cmd_ss >> bfname;
						b_enable = true;
					} else if (token == "-d") {
						cmd_ss >> duration;
					} else if (token == "-w") {
						cmd_ss >> warmup;
					} else if (token == "-p") {
						cmd_ss >> parallel_factor;
					} else if (token == "-i") {
						cmd_ss >> send_interval;
					} else if (token == "-s") {
						string start;
						cmd_ss >> start;
						query = cmd.substr(cmd.find(start));
						q_enable = true;
						break ;
					} else {
						if (IS_MASTER(proxy)) {
							cout << "Unknown option: " << token << endl;
							print_help();
						}
						goto next;
					}
				}

				if (f_enable) {
					// use the main proxy thread to run a single query
					if (IS_MASTER(proxy)) {
						ifstream ifs(fname);
						if (!ifs) {
							cout << "Query file not found: " << fname << endl;
							continue ;
						}
						Logger logger;
						proxy->run_single_query(ifs, cnt, logger);
						logger.print_latency(cnt);
					}
				}

				if (b_enable) {
					Logger logger;

                    if (duration <= 0 || warmup < 0 || parallel_factor <= 0) {
                        cout << "[ERROR] invalid parameters for batch mode! "
                             << "(duration=" << duration << ", warmup=" << warmup
                             << ", parallel_factor=" << parallel_factor << ")" << endl;
                        continue;
                    }

                    if (duration <= warmup) {
                        cout << "Duration time (" << duration
                             << "sec) is less than warmup time ("
                             << warmup << "sec)." << endl;
                        continue;
                    }

                    ifstream ifs(bfname);
                    if (!ifs) {
                        PRINT_ID(proxy);
                        cout << "Configure file not found: " << bfname << endl;
                        continue ;
                    }

					proxy->run_batch_query(ifs, duration, warmup, parallel_factor, send_interval, logger);
                    ifs.close();

					console_barrier(proxy->tid);

					// print a statistic of runtime for the batch processing on all servers
					if (IS_MASTER(proxy)) {
                        logger.set_figure(&fig_thpt_latency);

						for (int i = 0; i < global_num_servers * global_num_proxies - 1; i++) {
							Logger other = console_recv<Logger>(proxy->tid);
							logger.merge(other);
						}

                        cout << "-----------------------------------------------------" << endl;
                        cout << "Result of " << bfname << ":" << endl;
                        cout << "warmup: " << warmup << ", " << "duration: " << duration;
                        cout << ", parallel_factor: " << parallel_factor << ", send_interval: " << send_interval << endl;
                        logger.aggregate();
						logger.print_thpt();
                        cout << "-------------------- Figures Start ------------------" << endl;

                        logger.print_cdf();
                        logger.analyse();
                        logger.print_data();
                        cout << "-------------------- Figures End --------------------" << endl;

                        // proxy->notify_slaves(BATCH_SENTRY_FIN);
					} else {
						// send logs to the master proxy
						console_send<Logger>(0, 0, logger);
					}

				}

				if (q_enable) {
					// TODO: SPARQL string
					if (IS_MASTER(proxy)) {
						// TODO
						cout << "Query: " << query << endl;
						cout << "The option '-s' is unsupported now!" << endl;
					}
				}
			} else {
				if (IS_MASTER(proxy)) {
					cout << "Unknown command: " << token << endl;
					print_help();
				}
			}
		}
	}
}
