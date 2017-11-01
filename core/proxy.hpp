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

#include "config.hpp"
#include "coder.hpp"
#include "query.hpp"
#include "adaptor.hpp"
#include "parser.hpp"
#include "string_server.hpp"
#include "logger.hpp"

#include "mymath.hpp"
#include "timer.hpp"

#include "planner.hpp"
#include "data_statistic.hpp"

volatile int recv_large_cnt = 0;

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

            if (type_reply.dummy)
                return false;

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


	Proxy(int sid, int tid, String_Server *str_server, Adaptor *adaptor, data_statistic *statistic )
		: sid(sid), tid(tid), str_server(str_server), adaptor(adaptor),
          coder(sid, tid), parser(str_server), statistic(statistic) { }

	void setpid(request_or_reply &r) { r.pid = coder.get_and_inc_qid(); }

	void send_request(request_or_reply &r) {
		assert(r.pid != -1);

		if (r.start_from_index()) {
            printf("INFO#%d: [Proxy] large query!\n", sid);
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
        if (r.dummy)
            return r;

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
		if (success && r.start_from_index()) {
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



	void blocking_run_large_query(istream &is, Logger &logger) {
		int ntypes;
		int nqueries;
		int try_round = 1;
        int nlarge_types = 0;
		int send_cnt = 0, recv_cnt = 0, nsl_cnt = 0;

        assert(global_enable_large_query == true);

        is >> ntypes >> nqueries >> try_round >> nlarge_types;

        int nsmall_types = ntypes - nlarge_types;
        vector<request_template> tpls(nsmall_types);
        vector<request_or_reply> nsl_reqs(nlarge_types);
        vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;
            // skip selective queries
            if (i < nsmall_types)
                continue;

			bool success = parser.parse(ifs, nsl_reqs[i - nsmall_types]);
            ifs.close();

			if (!success) {
				cout << "ERROR: sparql parse error" << endl;
				return ;
			}
		}

        proxy_barrier(tid);

		logger.init(ntypes);

        // nqueries = (int)(nqueries * (0.05 / global_num_servers_for_large_query));
        nqueries = global_num_heavy_queries / global_num_servers;
        for (int i = 0; i < nqueries; i++) {
            int idx = nsmall_types + i % nlarge_types;
            request_or_reply request;

            // generate large query
            request = nsl_reqs[idx - nsmall_types];
            request.comp_dev = GPU_comp;
            nsl_cnt ++;
            setpid(request);

            request.blind = true; // avoid send back results by default
            logger.start_record(request.pid, idx);
            send_request(request);
            send_cnt ++;

            // for non-selective query, we must receive the reply before sending other queries
            request_or_reply r;
            r = recv_reply();
            assert(r.pid == request.pid);
            recv_cnt++;
            recv_large_cnt++;
            logger.end_record(r.pid);
            if (send_cnt > 0 && send_cnt % 50 == 0)
                printf("Blocking Proxy[sid:%d,tid:%d] has sent %d queries.\n", sid, tid, send_cnt);
        }

done:
        printf("Blocking Proxy[sid:%d,tid:%d] finish its work. send_cnt: %d, recv_cnt: %d, nsl_cnt: %d\n", sid, tid, send_cnt, recv_cnt, nsl_cnt);
        logger.finish();

        assert(recv_large_cnt == nqueries);

	}


	void nonblocking_run_batch_query1(istream &is, Logger &logger) {
		int ntypes;
		int nqueries;
		int try_round = 1;
        int nlarge_types = 0;
		int send_cnt = 0, recv_cnt = 0, nsl_cnt = 0;

        assert(global_enable_large_query == true);
        is >> ntypes >> nqueries >> try_round >> nlarge_types;

        int nsmall_types = ntypes - nlarge_types;
        vector<request_template> tpls(nsmall_types);
        vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

            // skil non-selective queries
            if (i >= nsmall_types)
                continue;

			bool success = parser.parse_template(ifs, tpls[i]);
            ifs.close();

			if (!success) {
				cout << "ERROR: Template parse failed!" << endl;
				return;
			}

            success = fill_template(tpls[i]);
            if (!success) {
                printf("Proxy[sid:%d,tid:%d] >>>>>>>>>> Timeout from fill_template!.\n", sid, tid);
                goto done;
            }
		}

        proxy_barrier(tid);

		logger.init(ntypes);
        int nlarge = global_num_heavy_queries / global_num_servers;
        while (recv_large_cnt < nlarge) {

			for (int t = 0; t < global_parallel_factor; t++) {
                int idx = mymath::get_distribution(coder.get_random(), loads);
                // only generate selective queries
                while (idx >= nsmall_types)
                    idx = mymath::get_distribution(coder.get_random(), loads);

                request_or_reply request;
                request = tpls[idx].instantiate(coder.get_random());
                assert(request.comp_dev == CPU_comp);
                setpid(request);
                request.blind = true; // avoid send back results by default
                logger.start_record(request.pid, idx);

                send_request(request);
                send_cnt ++;
                if (send_cnt > 0 && send_cnt % 5000 == 0)
                    printf("Proxy[sid:%d,tid:%d] has sent %d queries.\n", sid, tid, send_cnt);
			}

			// wait a piece of time and try several times
			for (int i = 0; i < try_round; i++) {
				timer::cpu_relax(100);

				// try to recieve the replies (best of effort)
				request_or_reply r;
				bool success = tryrecv_reply(r);
				while (success) {
                    recv_cnt ++;
                    logger.end_record(r.pid);
					success = tryrecv_reply(r);
				}
			}
		}

done:
        printf("Proxy[sid:%d,tid:%d] finish its work. send_cnt: %d, recv_cnt: %d, nsl_cnt: %d\n", sid, tid, send_cnt, recv_cnt, nsl_cnt);
        logger.finish();
	}

	void nonblocking_run_batch_query2(istream &is, Logger &logger) {
		int ntypes;
		int nqueries;
		int try_round = 1;
        int nlarge_types = 0;
		int send_cnt = 0, recv_cnt = 0, nsl_cnt = 0;

        assert(global_enable_large_query == false);
        is >> ntypes >> nqueries >> try_round >> nlarge_types;


        int nsmall_types = ntypes - nlarge_types;
        vector<request_template> tpls(nsmall_types);
        vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

            // skil non-selective queries
            if (i >= nsmall_types)
                continue;

			bool success = parser.parse_template(ifs, tpls[i]);
            ifs.close();

			if (!success) {
				cout << "ERROR: Template parse failed!" << endl;
				return;
			}

            success = fill_template(tpls[i]);
            if (!success) {
                printf("Proxy[sid:%d,tid:%d] >>>>>>>>>> Timeout from fill_template!.\n", sid, tid);
                goto done;
            }
		}

        proxy_barrier(tid);

		logger.init(ntypes);
        while (recv_cnt < nqueries) {

			for (int t = 0; t < global_parallel_factor; t++) {

                int idx = mymath::get_distribution(coder.get_random(), loads);
                // only generate selective queries
                while (idx >= nsmall_types)
                    idx = mymath::get_distribution(coder.get_random(), loads);

                request_or_reply request;
                request = tpls[idx].instantiate(coder.get_random());
                assert(request.comp_dev == CPU_comp);
                setpid(request);
                request.blind = true; // avoid send back results by default
                logger.start_record(request.pid, idx);

                send_request(request);
                send_cnt ++;
                if (send_cnt > 0 && send_cnt % 5000 == 0)
                    printf("Proxy[sid:%d,tid:%d] has sent %d queries.\n", sid, tid, send_cnt);
			}

			// wait a piece of time and try several times
			for (int i = 0; i < try_round; i++) {
				timer::cpu_relax(100);

				// try to recieve the replies (best of effort)
				request_or_reply r;
				bool success = tryrecv_reply(r);
				while (success) {
                    recv_cnt ++;
                    logger.end_record(r.pid);
					success = tryrecv_reply(r);
				}
			}
		}

done:
        printf("Proxy[sid:%d,tid:%d] finish its work. send_cnt: %d, recv_cnt: %d, nsl_cnt: %d\n", sid, tid, send_cnt, recv_cnt, nsl_cnt);
        logger.finish();
	}


    void run_batch_query(istream &is, Logger &logger) {
        int ntypes;
        int nqueries;
        int try_round = 1; // dummy

        is >> ntypes >> nqueries >> try_round;

        vector<int> loads(ntypes);
        vector<request_template> tpls(ntypes);

        // prepare various temples
        for (int i = 0; i < ntypes; i++) {
               string fname;
               is >> fname;
               ifstream ifs(fname);
               if (!ifs) {
                       cout << "Query file not found: " << fname << endl;
                       return ;
               }

               int load;
               is >> load;
               assert(load > 0);
               loads[i] = load;

               bool success = parser.parse_template(ifs, tpls[i]);
               if (!success) {
                       cout << "sparql parse error" << endl;
                       return ;
               }
               fill_template(tpls[i]);
        }

        logger.init();
        // send global_parallel_factor queries and keep global_parallel_factor flying queries
        for (int i = 0; i < global_parallel_factor; i++) {
               int idx = mymath::get_distribution(coder.get_random(), loads);
               request_or_reply r = tpls[idx].instantiate(coder.get_random());

               setpid(r);
               r.blind = true;  // avoid send back results by default
               logger.start_record(r.pid, idx);
               send_request(r);
        }

           // recv one query, and then send another query
        for (int i = 0; i < nqueries - global_parallel_factor; i++) {
               // recv one query
               request_or_reply r2 = recv_reply();
               logger.end_record(r2.pid);

               // send another query
               int idx = mymath::get_distribution(coder.get_random(), loads);
               request_or_reply r = tpls[idx].instantiate(coder.get_random());

               setpid(r);
               r.blind = true;  // avoid send back results by default
               logger.start_record(r.pid, idx);
               send_request(r);
        }

        // recv the rest queries
        for (int i = 0; i < global_parallel_factor; i++) {
               request_or_reply r = recv_reply();
               logger.end_record(r.pid);
        }

        logger.finish();
        logger.print_thpt();

    }




};
