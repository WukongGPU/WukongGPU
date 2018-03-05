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

#include <mpi.h>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <signal.h>
#include <sys/time.h>
#include <map>
#include <unistd.h>

#include "config.hpp"
#include "coder.hpp"
#include "query.hpp"
#include "adaptor.hpp"
#include "parser.hpp"
#include "string_server.hpp"
#include "logger.hpp"

#include "mymath.hpp"
#include "timer.hpp"
#include "mem.hpp"

#include "planner.hpp"
#include "data_statistic.hpp"

static void proxy_barrier(int tid)
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

// a vector of pointers of all local proxies
class Proxy;
std::vector<Proxy *> proxies;

class Proxy {

private:
    map<string, vector<int>> type_cache;

    Mem *mem;

    /**
     * @return false for timeout
     */
    bool fill_template(request_template &req_template) {
		req_template.ptypes_grp.resize(req_template.ptypes_str.size());
		for (int i = 0; i < req_template.ptypes_str.size(); i++) {
			string type = req_template.ptypes_str[i]; // the Types of random-constant

            auto it = type_cache.find(type);
            if (it != type_cache.end()) {
                req_template.ptypes_grp[i] = it->second;
                cout << "cache hit: " << type << " has " << req_template.ptypes_grp[i].size() << " candidates" << endl;
                continue;
            }

			request_or_reply type_request, type_reply;

			// a TYPE query to collect constants with the certain type
			if (!parser.add_type_pattern(type, type_request)) {
				cout << "ERROR: failed to add a special type pattern (type: "
				     << type << ")." << endl;
				assert(false);
			}

			// do a TYPE query to collect all of candidates for a certain type
			setpid(type_request);
			type_request.blind = false; // must take back the results
            assert(type_request.comp_dev == CPU_comp);
			send_request(type_request);
			type_reply = recv_reply();

			vector<int> candidates(type_reply.result_table);
			// There is no candidate with the Type for a random-constant in the template
			// TODO: it should report empty for all queries of the template
			assert(candidates.size() > 0);

            // add to cache
            type_cache[type] = candidates;
			req_template.ptypes_grp[i] = candidates;

			cout << type << " has " << req_template.ptypes_grp[i].size() << " candidates" << endl;
		}

        return true;
	}


public:
	int sid;    // server id
	int tid;    // thread id

	String_Server *str_server;
	Adaptor *adaptor;

	Coder coder;
	Parser parser;
    Planner planner;
    data_statistic *statistic;


	Proxy(int sid, int tid, String_Server *str_server, Adaptor *adaptor, data_statistic *statistic, Mem *mem)
		: sid(sid), tid(tid), str_server(str_server), adaptor(adaptor),
          coder(sid, tid), parser(str_server), statistic(statistic), mem(mem) { }


	void setpid(request_or_reply &r) { r.pid = coder.get_and_inc_qid(); }

	void send_request(request_or_reply &r) {
		assert(r.pid != -1);

		if (r.start_from_index()) {
			for (int i = 0; i < global_num_servers; i++) {
                if(r.comp_dev == GPU_comp && global_default_use_gpu_handle) {
                    //if it should be calculated by GPU, send it to GPU worker
					adaptor->send(i, global_num_proxies + global_num_engines, r);
                } else {
				    for (int j = 0; j < global_mt_threshold; j++) {
					    r.tid = j;
                        //choose random sequencial threads for mt
						// adaptor->send(i, global_num_proxies + (coder.get_random()+j) % global_mt_threshold, r);
                        //alway choose first few threads
                        adaptor->send(i, global_num_proxies + j, r);
				    }
                }
			}
			return ;
		}

		// submit the request to a certain engine
		int start_sid = mymath::hash_mod(r.cmd_chains[0], global_num_servers);

		// random assign request to range partitioned engines
		// NOTE: the partitioned mapping has better tail latency in batch mode
		int ratio = global_num_engines / global_num_proxies;
		int start_tid = 0;
        if (r.comp_dev == GPU_comp)
            start_tid = global_num_proxies + global_num_engines;
        else
            start_tid = global_num_proxies + (ratio * tid) + (coder.get_random() % ratio);
        adaptor->send(start_sid, start_tid, r);
	}

	request_or_reply recv_reply(void) {
		request_or_reply r = adaptor->recv();

		if (r.start_from_index()) {
            int mt_threshold = (r.comp_dev == GPU_comp ? 1 : global_mt_threshold);

			for (int count = 0; count < global_num_servers * mt_threshold - 1 ; count++) {
				request_or_reply r2 = adaptor->recv();
                assert(r.pid == r2.pid);
                assert(r2.start_from_index());
				r.row_num += r2.row_num;
				int new_size = r.result_table.size() + r2.result_table.size();
				r.result_table.reserve(new_size);
				r.result_table.insert(r.result_table.end(), r2.result_table.begin(), r2.result_table.end());
			}
		}
		return r;
	}

	bool tryrecv_reply(request_or_reply &r) {
		bool success = adaptor->tryrecv(r);
		if (success && r.start_from_index() && false) {
            // TODO: avoid parallel submit for try recieve mode
            cout << "Unsupport try recieve parallel query now!" << endl;
            assert(false);
		}

		return success;
	}

	void print_result(request_or_reply &r, int row2print) {
		cout << "The first " << row2print << " rows of results: " << endl;
		for (int i = 0; i < row2print; i++) {
			cout << i + 1 << ":  ";
			for (int c = 0; c < r.get_col_num(); c++) {
				int id = r.get_row_col(i, c);
				/*
				 * Must load the entire ID mapping files (incl. normal and index),
				 * If you want to print the query results with strings.
				 */
				if (str_server->id2str.find(id) == str_server->id2str.end())
					cout << id << "\t";
				else
					cout << str_server->id2str[r.get_row_col(i, c)] << "  ";
			}
			cout << endl;
		}
	}

	void run_single_query(istream &is, int cnt, Logger &logger) {
		request_or_reply request, reply;

        uint64_t t_parse1 =timer::get_usec();
		if (!parser.parse(is, request)) {
			cout << "ERROR: parse failed! ("
			     << parser.strerror << ")" << endl;
			return;
		}
        uint64_t t_parse2 =timer::get_usec();

        // populate required predicates of request
        for (int i = 0; (i + 3) < request.cmd_chains.size(); i += 4) {
            int pid = request.cmd_chains[i + 1];
            if (i==0 && request.start_from_index()) { // index_to_unknown
                request.preds.push_back(request.cmd_chains[i]);
                continue;
            }
            request.preds.push_back(pid);
        }


    if (global_enable_planner) {
      // planner
      uint64_t t_plan1 =timer::get_usec();
      bool exec = planner.generate_plan(request, statistic);
      uint64_t t_plan2 =timer::get_usec();
      cout << "parse time : " << t_parse2 - t_parse1 << " usec" << endl;
      cout << "plan time : " << t_plan2 - t_plan1 << " usec" << endl;
      if (exec == false) { // for empty result
        cout<< "(last) result size: 0" << endl;
        return ;
      }
    }
        //decide use GPU or CPU
        planner.choose_computing_device(request, statistic);
		logger.init();
		for (int i = 0; i < cnt; i++) {
			setpid(request);
			request.blind = true; // avoid send back results by default
			send_request(request);
			reply = recv_reply();
		}
		logger.finish();

		cout << "(last) result size: " << reply.row_num << endl;
		if (!global_silent && !reply.blind)
			print_result(reply, min(reply.row_num, global_max_print_row));
	}


    /* mixed light & heavy */
    inline bool is_fork_join(int idx) {
        return idx == 6 || idx == 9;
    }

    inline int idx_for_q2q3(void) {
        return (7 + coder.get_random() % 2);
    }
    inline int idx_for_q1q7(void) {
        static int arr[2] = {6, 9};
        return arr[ coder.get_random() % 2 ];
    }

	int run_batch_query(istream &is, int d, int w, int p, int i, Logger &logger) {
		uint64_t duration = SEC(d);
		uint64_t warmup = SEC(w);
        uint64_t send_interval = MSEC(i);
		int parallel_factor = p;
		int try_rounds = 5;

		int ntypes, nsmall_types, nlarge_types;
		is >> ntypes >> nlarge_types;

		if (ntypes <= 0) {
			cout << "[ERROR] invalid #query_types! (" << ntypes << " < 0)" << endl;
			return -2; // parsing failed
		}

        nsmall_types = ntypes - nlarge_types;
		vector<request_template> tpls(nsmall_types);
        vector<request_or_reply> nsl_reqs(nlarge_types);
		vector<int> loads(ntypes);

        // prepare work
		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "[ERROR] Query file not found: " << fname << endl;
				return -1; // file not found
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

			// bool success = parser.parse_template(ifs, tpls[i]);
            bool success = (i >= nsmall_types) ?
                    parser.parse(ifs, nsl_reqs[i - nsmall_types]):
                    parser.parse_template(ifs, tpls[i]);

            ifs.close();

			if (!success) {
				cout << "[ERROR] Template parsing failed!" << endl;
				return -2; // parsing failed
			}

            if (i < nsmall_types) {
                fill_template(tpls[i]);
            }
		}

		logger.init(nsmall_types, nlarge_types, parallel_factor);

        proxy_barrier(tid);

        if (sid == 0 && tid == 0) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  [MASTER]: start to work!\n");
        } else {
            printf("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[  [SLAVE]: start to work!\n");
        }

		bool timing = false;
		int send_cnt = 0, recv_cnt = 0,
            send_light_cnt = 0, send_heavy_cnt = 0,
            recv_light_cnt = 0, recv_heavy_cnt = 0,
            flying_light_cnt = 0, flying_heavy_cnt = 0;
        int start_thpt_cnt = 0, end_thpt_cnt = 0;
		uint64_t init = timer::get_usec();
		while ((timer::get_usec() - init) < duration) {
			// send requests
			for (int t = 0; t < parallel_factor; t++) {
				int idx = mymath::get_distribution(coder.get_random(), loads);
				request_or_reply request;

                if (idx >= nsmall_types) {  // large query
                    if (flying_heavy_cnt >= parallel_factor || flying_heavy_cnt >= global_heavy_flying_cnt)
                        continue;

                    request = nsl_reqs[idx - nsmall_types];
                    request.comp_dev = GPU_comp;

                } else {    // light query
                    if (flying_light_cnt >= parallel_factor)
                        continue;
                    request = tpls[idx].instantiate(coder.get_random());
                }

                if (flying_light_cnt >= parallel_factor && flying_heavy_cnt >= parallel_factor) {
                    printf("[ERROR] Proxy[%d:%d] send rate is too high!\n", sid, tid);
                    assert(false);
                }

				if (global_enable_planner)
					planner.generate_plan(request, statistic);

				setpid(request);
				request.blind = true; // always not take back results in batch mode

                if (request.start_from_index()) {
                    assert(request.comp_dev == GPU_comp);
                    logger.start_record(request.pid, idx, true);
                    send_heavy_cnt ++;
                } else {
				    logger.start_record(request.pid, idx);
                    send_light_cnt ++;
                }

				send_request(request);
				send_cnt++;
			}   // end send request loop

            // Siyuan: 要减慢发送的速度，那么就在receive这里做文章，
            // 比如我希望发送间隔是1s(1000ms)，那么在发完reqs之后，我就sleep(1ms)
            // wakeup之后去recv reply，然后继续sleep(1ms），醒来后去recv...
            // 一直循环到1s。
            uint64_t sleep_begin = timer::get_usec();
            uint64_t sleep_unit = send_interval > 0 ? MSEC(1) : 0;

            do {
                // receive replies (best of effort)
                for (int i = 0; i < try_rounds; i++) {
                    request_or_reply r;
                    while (tryrecv_reply(r)) {
                        if (!r.start_from_index())
                            recv_light_cnt ++;

                        recv_cnt = logger.end_record(r.pid);
                        recv_heavy_cnt = recv_cnt - recv_light_cnt;
                    }
                }

                usleep(sleep_unit);
            } while ((timer::get_usec() - sleep_begin) < send_interval);


            // for brevity, only print the timely thpt of master proxy.
            if (sid == 0 && tid == 0)
    			logger.print_timely_thpt(recv_light_cnt, recv_heavy_cnt);

			// requests during warmup not included in thpt calculation
			if (!timing && (timer::get_usec() - init) >= warmup) {
                start_thpt_cnt = recv_cnt;
				logger.start_calc_thpt(recv_light_cnt, recv_heavy_cnt, send_cnt);
                printf("Proxy[%d:%d]: start_thpt_cnt=%d\n", sid, tid, start_thpt_cnt);
				timing = true;
			}

            flying_light_cnt = send_light_cnt - recv_light_cnt;
            flying_heavy_cnt = send_heavy_cnt - recv_heavy_cnt;

		} // end main loop

		logger.end_calc_thpt(recv_light_cnt, recv_heavy_cnt, send_cnt);
        end_thpt_cnt = recv_cnt;

		// recieve all replies to calculate the tail latency
        printf("Proxy[%d:%d]: end_thpt_cnt=%d\n", sid, tid, end_thpt_cnt);
        while (recv_cnt < send_cnt) {
            request_or_reply r;
            while (tryrecv_reply(r)) {
                recv_cnt = logger.end_record(r.pid);
            }
        }

        logger.finish();
        printf("Proxy[%d:%d] Finished. send_cnt=%d, send_heavy_cnt=%d, send_light_cnt=%d, thpt_cnt=%d, tail_cnt=%d\n",
                sid, tid, send_cnt, send_heavy_cnt, send_light_cnt, (end_thpt_cnt - start_thpt_cnt), (recv_cnt - end_thpt_cnt));

		return 0; // success
	}

};
