# +
# # +
# # %load_ext autoreload
# # %autoreload 2

from functools import cache
import importlib
import scipy.special as ss
import functools
import xgi
import math
from collections import Counter, defaultdict
from multiprocessing import Pool
m = importlib.import_module(".model", "src")
sem = importlib.import_module(".sem", "src") 
em = importlib.import_module(".em", "src")
ll = importlib.import_module(".likelihoods", "src")
from matplotlib import pyplot as plt
import numpy as np
import pickle

#eta   = 0.7   # node retention in edges
#beta  = [0.71, 0.21, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
#gamma = [0.3, 0.2, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 
#         0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01, 
#         0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01]   # poisson number of nodes from graph

eta   = 0.2   # node retention in edges
beta  = [0.71, 0.21, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
gamma = [0.3, 0.2, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 
         0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01, 
         0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01]   # poisson number of nodes from graph


smaller_Hgraphs = ["congress-bills", "contact-high-school", "contact-primary-school", "dawn", "diseasome", 
                   "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", "invs13", "invs15", 
                   "kaggle-whats-cooking", "malawi-village", "ndc-classes", "ndc-substances", "science-gallery", 
                   "sfhh-conference"]
                                          
bigger_Hgraphs= ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "tags-ask-ubuntu", "tags-math-sx", 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]


realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "dawn", "diseasome","disgenenet", "email-enron", "email-eu", "hospital-lyon",
                 "hypertext-conference", "invs13", "invs15", "kaggle-whats-cooking", "malawi-village", "ndc-classes",
                 "ndc-substances", "science-gallery", "sfhh-conference","tags-ask-ubuntu", "tags-math-sx" , 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]

def initialize_graph_dict(e0, e1):
    
    H = m.GrowingHypergraph()
    H.add_edge(e0)
    H.add_edge(e1)
    
    r = defaultdict(int)
    intersect = len(e0.intersection(e1))
    r[(len(e0), len(e1), intersect)] = 1

    edge_size_dict = defaultdict(int)
    edge_size_dict[len(e0)] += 1
    edge_size_dict[len(e1)] += 1
    
    return H, r, edge_size_dict



def aggregate_esd(H, r, edge_size_dict, eid, e_):
    
    e_size = len(e_)    
    # find all neighbors that have intersection with e_
    de = H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)

    # count all the intersection sizes and add to r
    for nb in de:
        nb_size = len(nb)
        k = len(e_.intersection(nb))
        r[(e_size, nb_size, k)] += 1
        if nb_size != e_size:
            r[(nb_size, e_size, k)] += 1


    neighbor_sizes = [len(x) for x in de]
    nb_size_cnt = Counter(neighbor_sizes)
    for j, nj in edge_size_dict.items():
        assert(nj >= nb_size_cnt[j])
        r[(j, e_size, 0)] += nj - nb_size_cnt[j]
        if j != e_size:
            r[(e_size, j, 0)] += nj - nb_size_cnt[j]


    edge_size_dict[e_size] += 1
    
    return r, edge_size_dict 



def create_HG_rijk(dataset):
    
    # Load the real-world hypergraph and count the number of timestamps depending on the 
    # dataset = "email-enron"
    H1 = xgi.load_xgi_data(dataset)
    
    H1_edges = list(H1.edges)
    
    GH = m.GrowingHypergraph(H1)
    timesteps = GH.H.num_edges
    e0 = H1.edges.members(H1_edges[0])
    e1 = H1.edges.members(H1_edges[1])
    
    expected_from_graph = np.mean(H1.edges.size.aslist())
    
    H, H_r, H_esd = initialize_graph_dict(e0, e1)
    H_PA, H_PA_r, H_PA_esd = initialize_graph_dict(e0, e1)
    H_ER, H_ER_r, H_ER_esd = initialize_graph_dict(e0, e1)
    H_PA_exact, H_PA_exact_r, H_PA_exact_esd = initialize_graph_dict(e0, e1)
    H_ER_exact, H_ER_exact_r, H_ER_exact_esd = initialize_graph_dict(e0, e1)
    


    for cc in range(2, timesteps):
        new_edge = set(H1.edges.members(str(cc)))
        
        #print(new_edge)
        
        expected_new = len(new_edge -set(H1.nodes))
        exact_num = len(new_edge) - expected_new 
        
        eid = cc
        
        e_ = new_edge
        H.add_edge(e_)
        
        #print(H.edges)
        H_r, H_esd = aggregate_esd(H, H_r, H_esd, eid, e_)
        
        e_ = H_PA.sample_edge_alternative("PA", False, exact_num, expected_from_graph, expected_new)
        H_PA_r, H_PA_esd = aggregate_esd(H_PA, H_PA_r, H_PA_esd, eid, e_)
        
        e_ = H_ER.sample_edge_alternative("ER", False, exact_num, expected_from_graph, expected_new)
        H_ER_r, H_ER_esd = aggregate_esd(H_ER, H_ER_r, H_ER_esd, eid, e_)
        
        e_ = H_PA_exact.sample_edge_alternative("PA", True, exact_num, expected_from_graph, expected_new)
        H_PA_exact_r, H_PA_exact_esd= aggregate_esd(H_PA_exact, H_PA_exact_r, H_PA_exact_esd, eid, e_)
        
        e_ = H_ER_exact.sample_edge_alternative("ER", True, exact_num, expected_from_graph, expected_new)
        H_ER_exact_r, H_ER_exact_esd= aggregate_esd(H_ER_exact, H_ER_exact_r, H_ER_exact_esd, eid, e_)
        
        if cc%100 == 0:
            with open('simulation/' + dataset + '_orig_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_r, fp)
            with open('simulation/' + dataset + '_PA_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_PA_r, fp)
            with open('simulation/' + dataset + '_ER_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_ER_r, fp)
            with open('simulation/' + dataset + '_PA_exact_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_PA_exact_r, fp)
            with open('simulation/' + dataset + '_ER_exact_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_ER_exact_r, fp)
# -

with Pool(len(realworld_Hgraphs)) as p:
    print(p.map(create_HG_rijk, realworld_Hgraphs))

# +

# dataset = "email-enron"
# #H1 = xgi.load_xgi_data(dataset)
# create_HG_rijk(dataset)
# #H1.edges.members
# #list(H1.edges)
