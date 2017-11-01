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

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

#include "query.hpp"
#include "string_server.hpp"

#include "SPARQLParser.hpp"

using namespace std;

inline bool is_upper(string str1, string str2) {
    return boost::to_upper_copy<std::string>(str1) == str2;
}

static string read_input(istream& in)
  // Read a stream into a string
{
  string result;
  while (true) {
    string s;
    getline(in,s);
    result+=s;
    if (!in.good())
      break;
    result+='\n';
  }
  return result;
}

/**
 * Q := SELECT RD WHERE GP
 *
 * The types of tokens (supported)
 * 0. SPARQL's Prefix e.g., PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
 * 1. SPARQL's Keyword (incl. SELECT, WHERE)
 *
 * 2. pattern's constant e.g., <http://www.Department0.University0.edu>
 * 3. pattern's variable e.g., ?X
 * 4. pattern's random-constant e.g., %ub:GraduateCourse (extended by Wukong in batch-mode)
 *
 */
class Parser {
private:
    const static int64_t PTYPE_PH = (INT64_MIN + 1); // place holder of pattern type (a special group of objects)
    const static int64_t DUMMY_ID = (INT64_MIN);

    // str2ID mapping for pattern constants (e.g., <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> 1)
    String_Server *str_server;

    // str2ID mapping for pattern variables (e.g., ?X -1)
    boost::unordered_map<string, int64_t> pvars;

    // abbr2str mapping for prefixes (e.g., rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
    boost::unordered_map<string, string> prefixes;


    request_template req_template;

    // support at most one fetch-result optimization
    int corun_step;
    int fetch_step;

    void clear(void) {
        prefixes.clear();
        pvars.clear();

        req_template = request_template();
        valid = true;
        fetch_step = -1;
        corun_step = -1;
    }

    vector<string> get_tokens(istream &is) {
        vector<string> tokens;
        string t;
        while (is >> t)
            tokens.push_back(t);
        return tokens;
    }

    // Siyuan: SPARQL的"select"语句中，"where"必须要大写，否则extract()里的idx会越界
    bool extract(vector<string> &tokens) {
        int idx = 0;

        // prefixes (e.g., PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
        while (tokens.size() > idx && tokens[idx] == "PREFIX") {
            if (tokens.size() < idx + 3) {
                valid = false;
                strerror = "Invalid PREFIX";
                return valid;
            }

            prefixes[tokens[idx + 1]] = tokens[idx + 2];
            idx += 3;
        }

        /// TODO: support more (extended) clauses (e.g., PROCEDURE)

        // SELECT clause
        if ((tokens.size() > idx) && (tokens[idx++] != "SELECT")) {
            valid = false;
            strerror = "Invalid keyword";
            return valid;
        }

        /// TODO: result description (e.g., ?X ?Z)
        while ((tokens.size() > idx) && (tokens[idx++] != "WHERE"));

        if (tokens[idx++] != "{") {
            valid = false;
            strerror = "Invalid bracket";
            return valid;
        }

        // triple-patterns in WHERE clause
        vector<string> patterns;
        while (tokens[idx] != "}") {
            // CORUN and FETCH are two extend keywork by Wukong to support
            // collaborative execution. Different to fork-join execution,
            // the co-run execution will not send full-history. The patterns
            // within CORUN and FETCH will be executed on remote workers separatly
            // the results will be fetched back in the end.

            // Since they are not patterns, we just record the range of patterns.
            if (tokens[idx] == "CORUN")
                corun_step = patterns.size() / 4;
            else if (tokens[idx] == "FETCH")
                fetch_step = patterns.size() / 4;
            else
                patterns.push_back(tokens[idx]);
            idx++;
        }

        // 4-element tuple for each pattern
        // e.g., ?Y rdf:type ub:University .
        if (patterns.size() % 4 != 0) {
            valid = false;
            strerror = "Invalid pattern";
            return valid;
        }

        tokens.swap(patterns);
        return true;
    }

    /* The abbreviated forms in the SPARQL syntax are resolved to produce absolute IRIs */
    void resolve(vector<string> &tokens) {
        for (int i = 0; i < tokens.size(); i++) {
            for (auto iter : prefixes) {
                if (tokens[i].find(iter.first) == 0) {
                    string s = iter.second;
                    s.insert(s.find("#") + 1,
                             tokens[i],
                             iter.first.size(),
                             string::npos);
                    tokens[i] = s;
                    break;
                } else if (tokens[i][0] == '%'
                           && tokens[i].find(iter.first) == 1) {
                    // random-constants (start with '%') with a certain type,
                    // which is extended by Wukong in batch-mode
                    // e.g., %ub:University (incl. <http://www.Department0.University0.edu>, ..)
                    string s = "%" + iter.second;
                    s.insert(s.find("#") + 1,
                             tokens[i], iter.first.size() + 1,
                             string::npos);
                    tokens[i] = s;
                    break;
                }
            }
        }
    }

    int64_t token2id(string &token) {
        if (token[0] == '?') {  // pattern variable
            if (pvars.find(token) == pvars.end()) {
                // use negatie ID for variable
                int64_t id = - (pvars.size() + 1); // starts from -1
                pvars[token] = id;
            }
            return pvars[token];
        } else if (token[0] == '%') {  // pattern random-constant (batch mode)
            req_template.ptypes_str.push_back(token.substr(1));
            return PTYPE_PH;
        } else {  // pattern constant
            if (str_server->str2id.find(token) == str_server->str2id.end()) {
                strerror = "Unknown constant: " + token;
                valid = false;
                return DUMMY_ID;
            }
            return str_server->str2id[token];
        }
    }

    void dump_cmd_chains(void) {
        cout << "cmd_chain size: " << req_template.cmd_chains.size() << endl;
        for (int i = 0; i < req_template.cmd_chains.size(); i += 4) {
            cout << "pattern#" << i / 4 << ": "
                 << req_template.cmd_chains[i] << "\t"
                 << req_template.cmd_chains[i + 1] << "\t"
                 << req_template.cmd_chains[i + 2] << "\t"
                 << req_template.cmd_chains[i + 3] << "\t"
                 << endl;
        }
    }

    bool do_parse(vector<string> &tokens) {
        if (!extract(tokens))
            return false;

        resolve(tokens);

        // generate ID-format patterns
        for (int i = 0; (i + 3) < tokens.size(); i += 4) {
            // SPO
            string triple[3] = {tokens[i + 0], tokens[i + 1], tokens[i + 2]};

            dir_t d;
            if (tokens[i + 3] == "." || tokens[i + 3] == "->") {
                d = OUT;
            } else if (tokens[i + 3] == "<-") {
                d = IN;
                swap(triple[0], triple[2]);
            } else {
                valid = false;
                strerror = "Invalid seperator";
                return valid;
            }

            req_template.cmd_chains.push_back(token2id(triple[0]));
            req_template.cmd_chains.push_back(token2id(triple[1]));
            req_template.cmd_chains.push_back(d);
            req_template.cmd_chains.push_back(token2id(triple[2]));
        }

        // insert a new CORUN pattern
        if (fetch_step >= 0) {
            vector<int64_t> corun_pattern;
            corun_pattern.push_back((int64_t)DUMMY_ID); // unused
            corun_pattern.push_back((int64_t)DUMMY_ID); // unused
            corun_pattern.push_back(CORUN);
            corun_pattern.push_back(fetch_step + 1); // because we insert a new cmd in the middle

            req_template.cmd_chains.insert(req_template.cmd_chains.begin() + corun_step * 4,
                                           corun_pattern.begin(), corun_pattern.end());
        }

        // record positions of patterns with random-constants (batch mode)
        for (int i = 0; i < req_template.cmd_chains.size(); i++)
            if (req_template.cmd_chains[i] == PTYPE_PH)
                req_template.ptypes_pos.push_back(i);

        //dump_cmd_chains();
        return valid;
    }

    //_H_ means helper
    boost::unordered_map<unsigned, int64_t> _H_incVarIdMap;
    int64_t varId = -1;
    int64_t _H_inc_var_id(unsigned ori_id){
      if(_H_incVarIdMap.find(ori_id) == _H_incVarIdMap.end()){
        _H_incVarIdMap[ori_id] = varId;
        return varId--;
      }else{
        return _H_incVarIdMap[ori_id];
      }
    }
    int64_t _H_encode(const SPARQLParser::Element& element) {//const
      switch (element.type) {
        case SPARQLParser::Element::Variable:
          return _H_inc_var_id(element.id);
        case SPARQLParser::Element::Literal:
          cout<< "Not Support Literal" << endl;
          return DUMMY_ID;
        case SPARQLParser::Element::IRI:
          {
            string strIRI = "<"+element.value+">" ;
            if (str_server->str2id.find(strIRI) == str_server->str2id.end()) {
              cout<< "Unknown IRI: " + strIRI << endl;
              return DUMMY_ID;
            }
            return str_server->str2id[strIRI];
          }
        default:
          return DUMMY_ID;
      }
      return DUMMY_ID;
    }
    void _H_simplist_transfer(const SPARQLParser & parser, request_or_reply &r) {
      vector<int64_t> temp_cmd_chains ;
      SPARQLParser::PatternGroup group = parser.getPatterns();
      for (std::vector<SPARQLParser::Pattern>::const_iterator iter=group.patterns.begin(),limit=group.patterns.end();iter!=limit;++iter) {
        temp_cmd_chains.push_back(_H_encode(iter->subject));
        temp_cmd_chains.push_back(_H_encode(iter->predicate));
        temp_cmd_chains.push_back(OUT);
        temp_cmd_chains.push_back(_H_encode(iter->object));
      }
      r.cmd_chains = temp_cmd_chains;
    }

    bool _H_do_parse(istream & is, request_or_reply &r){
      string query = read_input(is);
      SPARQLLexer lexer(query);
      SPARQLParser parser(lexer);
      varId = -1;
      _H_incVarIdMap.clear();
      try {
         parser.parse();//sparql -f query/lubm_q1
         _H_simplist_transfer(parser,r);
      } catch (const SPARQLParser::ParserException& e) {
         cerr << "parse error: " << e.message << endl;
         return false;
      }
      return true;
    }

public:
    // the stat of query parsing
    bool valid;
    std::string strerror;

    Parser(String_Server *_ss): str_server(_ss) { clear(); }

    /* Used in single-mode */
    bool parse(istream &is, request_or_reply &r) {
      if (global_enable_planner) {
        // clear state of parser before a new parsing
        clear();
        //----------------------------------
        //printf("_H_do_parse(is) start\n");
        if(_H_do_parse(is,r)){
            //cout<<"------------------------------_H_do_parse(is)    success---------------------------------"<<endl;
            cout << "parsing triples is finished." << endl;

            return true;
        }
        is.clear();
        is.seekg(0);
        //printf("_H_do_parse(is) end\n");
        //----------------------------------

        assert(false);
      }

      clear();

        // spilt stream into tokens
        vector<string> tokens = get_tokens(is);

        // parse the tokens
        if (!do_parse(tokens))
            return false;

        if (req_template.ptypes_pos.size() != 0) {
            cout << "ERROR: request with PTYPE_PH." << endl;
            return false;
        }

        r.cmd_chains = req_template.cmd_chains;



        return true;
    }

    /* Used in batch-mode */
    bool parse_template(istream &is, request_template &r) {
        // clear state of parser before a new parsing
        clear();

        vector<string> tokens = get_tokens(is);
        if (!do_parse(tokens))
            return false;

        if (req_template.ptypes_pos.size() == 0) {
            cout << "ERROR: request_template without PTYPE_PH" << endl;
            return false;
        }

        r = req_template;


        return true;
    }

    bool add_type_pattern(string type, request_or_reply &r) {
        clear();
        r = request_or_reply();

        // add an additonal pattern cmd to collect pattern constants with a certain type
        r.cmd_chains.push_back(str_server->str2id[type]); // type ID
        r.cmd_chains.push_back(TYPE_ID);  // reserved ID for "rdf:type"
        r.cmd_chains.push_back(IN);
        r.cmd_chains.push_back(-1);
        return true;
    }

};
