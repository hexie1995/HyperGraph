# + endofcell="--"
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


timesteps = int(1e6)

H = m.GrowingHypergraph()
H.add_edge((0, 1))
H.add_edge((2, 3))


# this initialize r, the size of r should always be +1 to the maximum edge size. 
max_size = 3
# r = np.zeros(shape=(max_size, max_size, max_size))
r = defaultdict(int)
r[(2, 2, 0)] = 1
save_r = []
edge_size_dict = defaultdict(int)
edge_size_dict[2] += 2




for cc in range(timesteps):
    e_ = H.sample_edge(eta, gamma, beta, True)
    eid = H.num_edges-1
    e_size = len(e_)    
    # update the size of r accordingly. it should always be max_size + 1
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
    
    if cc%100 == 0:
        with open('simulation_2/r_{}.pkl'.format(cc), 'wb') as fp:
            pickle.dump(r, fp)


#with Pool(len(realworld_Hgraphs)) as p:
#    print(p.map(run_real_world, realworld_Hgraphs))

# # +
# # IF YOU WANT TO RUN THE BENCHMARKING DATASET
# for data_ in bench_mark:
#     run_bench_mark(data_)
# -
# --

# +
# # IF YOU WANT TO RUN THE BENCHMARKING DATASET
# for data_ in bench_mark:
#     run_bench_mark(data_)
# -

