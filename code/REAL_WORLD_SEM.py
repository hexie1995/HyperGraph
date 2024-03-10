# +
# %load_ext autoreload
# %autoreload 2

import importlib 
m = importlib.import_module(".model", "src")
sem = importlib.import_module(".sem", "src") # my implementation of baby SEM
em = importlib.import_module(".em", "src")
ll = importlib.import_module(".likelihoods", "src")
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import xgi
import copy
import random
import statistics
import matplotlib.pyplot as plt
import scipy.special as ss
import json
import os
import pickle

# init_pars = {"eta"   : 0.3, 
#              "beta"  : 0.7, 
#              "gamma" : [1/35]*35}



# track the parameter estimates over time

def experiment(H, steps = 5000, ax = None, batch_size = 30, verbose = False):
    
    #if not ax: 
    #    fig, ax = plt.subplots(1, 1)
        
    EM = sem.SEM(H, 
             ll.guaranteed_edge_sample_likelihood,
             ll.nodes_from_hypergraph_likelihood,
             ll.novel_nodes_likelihood,
             memoize = False)
    # pars = init_pars.copy(), 
    
    epochs = int(steps / batch_size)
    
    ETA   = np.zeros(epochs)
    BETA  = np.zeros(epochs)
    GAMMA = []

    # main loop
    for i in range(epochs):

        if i % max(int(epochs/20), 1) == 0 and verbose:
            print("--------")
            print(f"Completed epoch {i}")
            print(EM.pars)
            
        EM.SEM_step(0.002*batch_size, batch_size = batch_size) # both the stochastic E and the M steps are in here
        ETA[i] = EM.pars["eta"]
        BETA[i] = EM.pars["beta"]
        GAMMA.append(EM.pars["gamma"])
    
    
    return ETA, BETA, GAMMA
    
    
    
def run_real_world(data_name):
    
    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    ETA, BETA, GAMMA = experiment(H0)
    
    pars = {"dataset": data_name, "eta": ETA,  "beta": BETA,  "gamma": GAMMA}
    
    with open('results/res_{}.pkl'.format(data_name + "_SEM_new"), 'wb') as fp:
        pickle.dump(pars, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

def run_bench_mark(data_name):
    
    Dir = r"../data//"
    with open(os.path.join(Dir, data_name + '_pos.pkl'), 'rb') as h: pos_edges = pickle.load(h)
        
    H0 = xgi.Hypergraph(pos_edges)
    H0 = m.GrowingHypergraph(H0)
    ETA, BETA, GAMMA = experiment(H0)
    
    pars = {"dataset": data_name, "eta": np.mean(ETA),  "beta": np.mean(BETA),  "gamma": np.mean(GAMMA, axis = 0)}
    
    with open('results/res_{}.json'.format(data_name + "_SEM_new"), 'w') as fp:
        json.dump(pars, fp)
    


bench_mark = ["iAF1260b", "iJO1366", "uspto"]

realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "dawn", "diseasome","disgenenet", "email-enron", "email-eu", "hospital-lyon",
                 "hypertext-conference", "invs13", "invs15", "kaggle-whats-cooking", "malawi-village", "ndc-classes",
                 "ndc-substances", "science-gallery", "sfhh-conference","tags-ask-ubuntu", "tags-math-sx" , 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]

# IF YOU WANT TO ONLY RUN ONE DATASET
#run_real_world("email-enron")

# IF YOU WANT TO RUN ALL XGI DATASET IN PARALLEL
with Pool(len(realworld_Hgraphs)) as p:
     print(p.map(run_real_world, realworld_Hgraphs))

# +
# # IF YOU WANT TO RUN THE BENCHMARKING DATASET
# for data_ in bench_mark:
#     run_bench_mark(data_)

# +
#run_real_world("email-enron")
