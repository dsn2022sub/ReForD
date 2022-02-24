

import numpy as np
import networkx as nx

#load data, a weighted undirected graph
def load_data(path):
    normal_stream_path=path+'/normal/filtered-stream/'
    normal_base_path=path+'/normal/filtered-base/'
    attack_stram_path=path+'/attack/filtered-stream/'
    attack_base_path=path+'/attack/filtered-base/'
    G = nx.Graph()
    for i in range(125):
        path1=normal_stream_path+'stream-wget-normal-'+str(i)+'.txt'
        data1 = np.loadtxt(path1, dtype=str).astype('str')
        for ite in data1:
            G.add_edge(int(ite[0]), int(ite[1]), weight=1)#我们目前的图不是weighted graph

        path2=normal_base_path+'base-wget-normal-'+str(i)+'.txt'
        data2 = np.loadtxt(path2, dtype=str).astype('str')
        for ite in data2:
            G.add_edge(int(ite[0]), int(ite[1]), weight=1)#我们目前的图不是weighted graph

        path3=attack_base_path+'base-wget-attack-'+str(i)+'.txt'
        data3 = np.loadtxt(path3, dtype=str).astype('str')
        for ite in data3:
            G.add_edge(int(ite[0]), int(ite[1]), weight=1)#我们目前的图不是weighted graph

        path4=attack_stram_path+'stream-wget-attack-'+str(i)+'.txt'
        data4 = np.loadtxt(path4, dtype=str).astype('str')
        for ite in data4:
            G.add_edge(int(ite[0]), int(ite[1]), weight=1)#我们目前的图不是weighted graph
    return G


def get_feature(G):
    #feature dictionary which format is {node i's id:Ni, Ei, Wi, λw,i}
    featureDict = {}
    nodelist = list(G.nodes)
    for ite in nodelist:
        featureDict[ite] = []
        #the number of node i's neighbor
        Ni = G.degree(ite)
        featureDict[ite].append(Ni)
        #the set of node i's neighbor
        iNeighbor = list(G.neighbors(ite))
        #the number of edges in egonet i
        Ei = 0
        #sum of weights in egonet i
        Wi = 0
        #the principal eigenvalue(the maximum eigenvalue with abs) of egonet i's weighted adjacency matrix
        Lambda_w_i = 0
        Ei += Ni
        egonet = nx.Graph()
        for nei in iNeighbor:
            Wi += G[nei][ite]['weight']
            egonet.add_edge(ite, nei, weight=G[nei][ite]['weight'])
        iNeighborLen = len(iNeighbor)
        for it1 in range(iNeighborLen):
            for it2 in range(it1+1, iNeighborLen):
                #if it1 in it2's neighbor list
                if iNeighbor[it1] in list(G.neighbors(iNeighbor[it2])):
                    Ei += 1
                    Wi += G[iNeighbor[it1]][iNeighbor[it2]]['weight']
                    egonet.add_edge(iNeighbor[it1], iNeighbor[it2], weight=G[iNeighbor[it1]][iNeighbor[it2]]['weight'])
        egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
        eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
        eigenvalue.sort()
        Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        featureDict[ite].append(Ei)
        featureDict[ite].append(Wi)
        featureDict[ite].append(Lambda_w_i)
    return featureDict
