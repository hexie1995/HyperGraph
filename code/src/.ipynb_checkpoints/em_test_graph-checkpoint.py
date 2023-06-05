# +
import xgi
import numpy as np
import math
import itertools
import time
from collections import Counter
from multiprocessing import Pool
from model import *
from em_core import *

def test_error_graph(parameters):
    
    #p2 = np.array([[0.6, 0.01], [0.01, 0.6]])
    #sizes2 = [25, 25]
    #H2 = xgi.uniform_HSBM(50, 2, p2, sizes2) # links
    
    #p3 = np.array([[0.6, 0.01], [0.01, 0.6]])
    #sizes2 = [25, 25]
    #H2 = xgi.uniform_HSBM(50, 3, p3, sizes3)
    # right now only 2 edges or 3 edges at a time
    # DCSBM
    
    # n number of nodes, m edges size, 
    # xgi.uniform_HSBM(50, 200, [0.]) more flexible
    # xgi.uniform_HPPM() fewer parameters
    # hypergraph planted partition model
    # 2 community
    H0 = xgi.random_hypergraph(50, [0.1, 0.01])
    eta,gamma,beta, ieta,igamma,ibeta = parameters
    
    eta_vec= []
    gamma_vec=[]
    beta_vec=[]
    
    H_temp = H0
    
    # test a hypergraph that 100 edges has been added since the initial state
    # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself
    C = []
    for _ in range(100):
        H1 = GrowingHypergraph(H_temp)
        e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)
        heta, hgamma, hbeta, correct = EM_update(e_, H_temp, ieta, igamma, ibeta, 100)
        eta_vec.append(heta)
        gamma_vec.append(hgamma)
        beta_vec.append(hbeta)
        H_temp = H1.H
        C.append(correct)
    
    
    with open("output_graph.txt", "a") as f:
        print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta, file=f)
        
    
    with open("correctness.txt", "a") as f:
        print(C, file=f)
# -



# +
test_eta = [x / 10.0 for x in range(1,9)]
test_gamma = [x / 20.0 for x in range(1,6)]
test_beta = [x for x in range(5)]

init_eta = [0.25, 0.5, 0.75, 1]
init_gamma = [0.05, 0.1, 0.15]
init_beta = [1,5]


wrapper_list = []

wrapper_list.append(test_eta)
wrapper_list.append(test_gamma)
wrapper_list.append(test_beta)
wrapper_list.append(init_eta)
wrapper_list.append(init_gamma)
wrapper_list.append(init_beta)

combs = []


for i in itertools.product(*wrapper_list):
    combs.append(i)


# +
# H0 = xgi.random_hypergraph(50, [0.1, 0.01])
# eta,gamma,beta, ieta,igamma,ibeta = combs[5]
# eta_vec= []
# gamma_vec=[]
# beta_vec=[]

# H_temp = H0

# # test a hypergraph that 100 edges has been added since the initial state
# # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself

# for _ in range(100):
#     H1 = GrowingHypergraph(H_temp)
#     e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)
#     heta, hgamma, hbeta,correct = EM_update(e_, H_temp, ieta, igamma, ibeta, 100)
#     eta_vec.append(heta)
#     gamma_vec.append(hgamma)
#     beta_vec.append(hbeta)
#     H_temp = H1.H
# correct
# -

for count in range(120):
    
    print(count*200, count*200+200)
    count = count+1

    sub_combs = combs[count*200:count*200+200]

    with Pool(len(sub_combs)) as p:
        print(p.map(test_error_graph, sub_combs))
    
    time.sleep(60)

# +
# errors = np.loadtxt("output_graph.txt")
# errors = errors[~np.isnan(errors).any(axis=1)]
# errors = np.absolute(errors)

# +
# np.mean(errors,axis=0)
