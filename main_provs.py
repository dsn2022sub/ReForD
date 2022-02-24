#!usr/bin/python
# -*- coding: utf-8 -*-
import json
import math
import os
from collections import deque
from multiprocessing import Pool
import argparse
import networkx as nx
from tqdm import tqdm

import camflow.entity as c_entity
import camflow.merger as c_merger
import camflow.parser as c_parser
import camflow.preparer as c_preparer
import s.entity as s_entity
import s.parser as s_parser
# import atop.atop_parser as a_parser
import utils.hasher as hasher
from networkx.algorithms.community import k_clique_communities
import walker


def find_community(graph, k):
    return list(k_clique_communities(graph, k))


def get_runnable_fn(dataset, mode, prepare_camflow=True, prepare_s=True):
    def runnable_fn(i):
        # camflow
        print(mode)
        camflow_log_file = f'data/{mode}/{dataset}-{mode}-{i}.log'
        preprocessed_file = f'{dataset}-preprocessed-{mode}-{i}.txt'
        base_graph_file = f'data/{mode}/base/base-{dataset}-{mode}-{i}.txt'
        stream_graph_file = f'data/{mode}/stream/stream-{dataset}-{mode}-{i}.txt'
        all_graph_file = f'data/{mode}/all-{dataset}-{mode}-{i}.txt'
        nodes_map_file = f'data/maps/nodes-{dataset}-{mode}-{i}.txt'
        edges_map_file = f'data/maps/edges-{dataset}-{mode}-{i}.txt'
        # atop
        # atop_log_file = f'data/atop/atop_log-{i}'
        # parsed_atop_log_file = f'data/atop/parsed-{dataset}-{mode}-atop-{i}.txt'
        # filtered_atop_log_file = f'data/atop/filtered-{dataset}-{mode}-atop-{i}.txt'

        s_log_file = f'data/{mode}/{dataset}-{mode}-s-{i}.log'
        parsed_s_log_file = f'data/ss/parsed-{dataset}-{mode}-s-{i}.txt'

        # avoid magic value
        jiffies = True

        # prepare camflow
        if prepare_camflow:
            c_preparer.run(camflow_log_file, preprocessed_file, jiffies)
            c_parser.run(preprocessed_file, base_graph_file, stream_graph_file, jiffies, nodes_map_file, edges_map_file)
            c_merger.run(base_graph_file, stream_graph_file, all_graph_file)
            print('done')
            # clean tmp file

            os.remove(preprocessed_file)

        # prepare s
        if prepare_s:
            s_parser.run(s_log_file, parsed_s_log_file)

        # get maps for graph building
        # in fact, this has been a graph
        # --- nodes_map ---
        # {
        #   unreset_node_id: reset_node_id
        # }
        # --- edges_map ---
        # {
        #   reset_src_node_id: [reset_dst_node_id, logical_timestamp][]
        # }
        with open(nodes_map_file, 'r', encoding='utf-8', errors='ignore') as nodes_map_fh:
            nodes_map = json.load(nodes_map_fh)
        # with open(edges_map_file, 'r', encoding='utf-8', errors='ignore') as edges_map_file:
        #     edges_map = json.load(edges_map_file)

        # get generalized entities
        camflow_entities = c_entity.get_entities(camflow_log_file)

        s_entities = s_entity.get_entities(parsed_s_log_file)

        # get core entities
        core_entities = set()
        print('generating core entity')
        coreset = dict()

        for s_entity in s_entities:
            s_ts, s_entity_name, table_name, *_ = s_entity
            if not table_name.startswith('oc'):
                continue

            if not s_entity_name.endswith('\"'):
                s_entity_name = '\"' + s_entity_name + '\"'
            for camflow_entity in camflow_entities:
                camflow_ts, camflow_entity_name, node_id = camflow_entity
                if not camflow_entity_name.endswith('\"'):
                    camflow_entity_name = '\"' + camflow_entity_name + '\"'
                # we observe that camflow will be slower for about 30-100us
                # so we use the max value 100us, i.e. 0.0001s
                # since we have generalized entities, we can simply use `==`
                if -1.5 <= camflow_ts - s_ts <= 1.5 and table_name != '' and table_name in str(camflow_entity_name):
                    hashed_node_id = hasher.generate_hash(node_id)
                    core_entities.add(hashed_node_id)
                    if table_name in coreset.keys():
                        coreset[table_name] += 1
                    else:
                        coreset[table_name] = 1
        print("length of core entities: ")
        print(len(core_entities))
        print(i, ':', len(s_entities), len(camflow_entities))
        print(coreset)

        # build the graph
        # since we observe that core entities do not have in-edges, we try to add a dumb node (node id -1)
        # and make all its out-edges with timestamp -1, which means not exist, and earlier than any edges
        cg = nx.MultiDiGraph([('-1', nodes_map[core_entity], {'logical_ts': -1}) for core_entity in core_entities])
        reversed_cg = nx.MultiDiGraph(
            [('-1', nodes_map[core_entity], {'logical_ts': -1}) for core_entity in core_entities])
        # cg = nx.MultiDiGraph()

        with open(all_graph_file, 'r', encoding='utf-8', errors='ignore') as all_map_fh:
            for edge in all_map_fh:
                parts = edge.strip().split(':')
                src, dst, src_type = parts[0].split()
                dst_type = parts[1]
                edge_type = parts[2]
                logical_ts = int(parts[-3])
                actual_ts = parts[-2]
                isnetwork = parts[-1]

                cg.add_edge(src, dst, **{
                    'src': src,
                    'dst': dst,
                    'src_type': src_type,
                    'dst_type': dst_type,
                    'edge_type': edge_type,
                    'logical_ts': logical_ts,
                    'actual_ts': actual_ts,
                    'isnetwork': isnetwork
                })
                reversed_cg.add_edge(src, dst, **{
                    'src': dst,
                    'dst': src,
                    'src_type': dst_type,
                    'dst_type': src_type,
                    'edge_type': edge_type,
                    'logical_ts': logical_ts,
                    'actual_ts': actual_ts,
                    'isnetwork': isnetwork
                })

        # get filtered subgraph
        # this is a empirical value
        depth = 10
        # we use set to avoid repeat now
        # TODO: maybe we can enhance the paths by use list
        subgraph_edges = set()
        # use BFS, start from the dumb node
        # the queue saves tuples, which consist of (node_id, last_ts)
        all_node = set()

        for next_n in cg.successors('-1'):
            X = walker.random_walks(cg, n_walks=10000, walk_len=550, p=0.5, q=4, start_nodes=[next_n])
            # rev_X = walker.random_walks(reversed_cg, n_walks=10000, walk_len=550,  p=0.5, q=4,start_nodes=[next_n])
            frequency = {}
            for x in X:
                tmp = list(set(x))
                for t in tmp:
                    if t not in frequency:
                        frequency[t] = 1
                    else:
                        frequency[t] += 1
            cnt = 700
            frequency = dict(sorted(frequency.items(), key=lambda e: e[1], reverse=True))
            # print(frequency)
            for key, value in frequency.items():
                if cnt >= 0 and value >= 7:
                    all_node.add(key)
                    cnt = cnt - 1

            '''
            rev_frequency = {}
            for x in rev_X:
                tmp = list(set(x))
                for t in tmp:
                    if t not in rev_frequency:
                        rev_frequency[t] = 1
                    else :
                        rev_frequency[t] += 1
            cnt = 700
            rev_frequency = dict(sorted(rev_frequency.items(), key=lambda e: e[1],reverse=True))
            for key ,value in rev_frequency.items():
                if cnt >= 0 and value >=7:
                    all_node.add(key)
                    cnt = cnt -1
            print(len(rev_frequency))
            '''
            for k in range(len(X)):
                for j in range(len(X[k])):
                    all_node.add(X[k][j])
                # all_node = list(all_node)
        all_node = [str(k) for k in all_node]
        print('your list length:')
        print('orignal nodes')
        # print(list(cg.nodes))

        SG = cg.subgraph(all_node)
        sg_edges = list(SG.edges())
        print('num of subgraph')
        print(len(sg_edges))
        # print(sg_edges)
        for j in range(len(sg_edges)):
            # print('add : ' +str(sg_edges[j][0])+" "+ str(sg_edges[j][1]))
            subgraph_edges.add((sg_edges[j][0], sg_edges[j][1]))
        # print(subgraph_edges)

        '''
        k=5
        print ("############# k-Clique: %d ################" % k)
        rst_com = find_community(undirected_cg,k)
        print ("Count of Community being found %d" % len(rst_com))
        node_set = set()
        for i in range(len(rst_com)):
            sub_com=list(rst_com[i])
            for k in sub_com:
                node_set.add(k)
        node_set = list(node_set)
        SG=cg.subgraph(node_set)
        sg_edges=list(SG.edges())
        for j in range(len(sg_edges)):
            subgraph_edges.add((sg_edges[j][0],sg_edges[j][1]))
        
        '''

        '''
        bfs_q = deque()
        bfs_q.append(('-1', '-1', -1))
        while bfs_q and depth:
            size = len(bfs_q)
            for _ in range(size):
                current_n, last_n, current_ts = bfs_q.popleft()
                # ignore the dumb node
                if current_n != '-1':
                    subgraph_edges.add((last_n, current_n))
                for next_n in cg.successors(current_n):
                    out_edges = cg[current_n][next_n]
                    # since this is a multi-graph, we may have many out-edges in the dict format
                    # so we should use .items() to traverse them
                    for _, out_edge in out_edges.items():
                        next_ts = out_edge['logical_ts']
                        if next_ts >= current_ts:
                            bfs_q.append((next_n, current_n, current_ts))
            depth -= 1
        '''
        # remove dumb nodes and sort by timestamp

        edges = []
        for src, dst in subgraph_edges:
            src_dst_edges = cg[src][dst]
            for _, edge in src_dst_edges.items():
                if edge['logical_ts'] >= 0:
                    edges.append(edge)
        edges.sort(key=lambda e: e['logical_ts'])
        print("length of edges")
        print(len(edges))
        start_time = edges[0]['logical_ts']
        end_time = edges[-1]['logical_ts']
        print('start_time:' + str(start_time))
        print('end_time:' + str(end_time))

        print(i)
        print(mode)

        # write for unicorn to analyse
        log_len = 0
        balance = 0
        if i == 0 and mode == 'normal':
            log_len = 62
        elif i == 1 and mode == 'normal':
            log_len = 63
            balance = 62
        if i == 2 and mode == 'normal':
            log_len = 62
            balance = 125
        elif i == 3 and mode == 'normal':
            log_len = 63
            balance = 187
        if i == 0 and mode == 'attack':
            # log_len = 25
            log_len = 25
            balance = 0
        elif i == 1 and mode == 'attack':
            log_len = 13
            balance = 0
        elif i == 2 and mode == 'attack':
            log_len = 12
            balance = 13
        print(log_len)
        time_interval = (end_time - start_time) / log_len
        print('time_interval:' + str(time_interval))
        base_ratio = 0.1
        all_base_graph_size = int(math.ceil(len(edges)) / log_len)
        print(all_base_graph_size)
        # all_graph = [edges[i:i + all_base_graph_size] for i in range(0, len(edges), all_base_graph_size)]
        all_graph = []
        for i in range(log_len):
            start = start_time + i * time_interval
            end = start_time + (i + 1) * time_interval
            tmp = []
            for edge in edges:
                if edge['logical_ts'] >= start and edge['logical_ts'] <= end:
                    tmp.append(edge)
            all_graph.append(tmp)

        for k in range(log_len):
            print(k + balance)
            filtered_base_graph_file = f'data/{mode}/filtered-base/base-{dataset}-{mode}-{k + balance}.txt'
            filtered_stream_graph_file = f'data/{mode}/filtered-stream/stream-{dataset}-{mode}-{k + balance}.txt'
            base_graph_size = int(math.ceil(len(all_graph[k]) * base_ratio))
            base_graph_fh = open(filtered_base_graph_file, 'w')
            stream_graph_fh = open(filtered_stream_graph_file, 'w')

            # calc node unseen
            nodes_seen = set()
            # reset node id
            new_id_map = dict()
            new_id = 0

            for idx, edge in enumerate(all_graph[k]):
                src = edge['src']
                dst = edge['dst']
                src_type = edge['src_type']
                dst_type = edge['dst_type']
                edge_type = edge['edge_type']
                logical_ts = edge['logical_ts']

                src_unseen = int(src not in nodes_seen)
                dst_unseen = int(dst not in nodes_seen)

                nodes_seen.add(src)
                nodes_seen.add(dst)

                if src_unseen:
                    new_id_map[src] = new_id
                    src = new_id
                    new_id += 1
                else:
                    src = new_id_map[src]
                if dst_unseen:
                    new_id_map[dst] = new_id
                    dst = new_id
                    new_id += 1
                else:
                    dst = new_id_map[dst]

                if idx < base_graph_size:
                    # print(str(src)+'+'+str(dst))
                    base_graph_fh.write(f'{src} {dst} {src_type}:{dst_type}:{edge_type}:{logical_ts}\n')
                else:
                    # print(str(src)+'+'+str(dst))
                    stream_graph_fh.write(
                        f'{src} {dst} {src_type}:{dst_type}:{edge_type}:{src_unseen}:{dst_unseen}:{logical_ts}\n')
            print(len(all_graph[k]))
            base_graph_fh.close()
            stream_graph_fh.close()

    return runnable_fn


# avoid cannot pickle problem
def runnable_normal_wrapper(i):
    return run_normal(i)


def runnable_attack_wrapper(i):
    return run_attack(i)


if __name__ == '__main__':
    # with Pool(processes=5) as pl:
    #     pl.map(runnable_normal_wrapper, list(range(125)))
    # with Pool(processes=5) as pl:
    #     pl.map(runnable_attack_wrapper, list(range(25)))
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='the dataset you want to use', required=True)
    args = parser.parse_args()
    print('the dataset you want to handle is :' + args.dataset)
    run_normal = get_runnable_fn(args.dataset, 'normal')
    run_attack = get_runnable_fn(args.dataset, 'attack')

    for i in tqdm(range(2)):
        runnable_normal_wrapper(i)
    for i in tqdm(range(1)):
        runnable_attack_wrapper(i)
