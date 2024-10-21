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
import time



# track the parameter estimates over time

def experiment(H, g_len, steps = 5000, ax = None, batch_size = 30, verbose = False):
    
    
    start = time.time()
    
    # use the largest edge size as the starting length for both beta and gamma. 
    init_pars = {"eta" : 0.5, "beta" : [1/g_len]*g_len, "gamma" : [1/g_len]*g_len}
    
    
    EM = sem.SEM(H, 
             ll.guaranteed_edge_sample_likelihood,
             ll.nodes_from_hypergraph_likelihood,
             ll.novel_nodes_likelihood,
             memoize = False,
             pars = init_pars.copy())
    
    
    # pars = init_pars.copy(), 
    if steps > 200000:
        steps = 200000
    
    epochs = int(steps / batch_size)
    
    ETA   = []
    BETA  = []
    GAMMA = []
    
    ADF =[]
    early_break = None
    current_eta = 100

    # main loop
    for i in range(epochs):
        
        step_size = 1/(i+batch_size+1)
        
        if step_size <1e-7:
            step_size = 1e-7
        
        # change the step size at each loop, linear to the steps. 
        EM.SEM_step(step_size*batch_size, batch_size = batch_size) # both the stochastic E and the M steps are in here
        ETA.append(EM.pars["eta"])
        BETA.append(EM.pars["beta"])
        GAMMA.append(EM.pars["gamma"])
        ADF.append(EM.pars["eta"])
        
        if i % 100 == 0 and i > 100:
            print("--------")
            print(f"calculating last averaged differences")
            result = np.mean(ADF[-100:])
            curr_diff = (abs(result - current_eta))/(current_eta)
            current_eta = EM.pars["eta"]
            print(curr_diff)
    
            if  curr_diff <0.01:
                T = i*batch_size
                early_break = True
                print("Converged after ", T, "steps")
                break
        
    if early_break is None:
        early_break = False
        T = steps
        
    end = time.time()
    time_takes = end - start
    
    
    return ETA, BETA, GAMMA, time_takes, T
    
    
    
def run_real_world(data_name):
    
    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    # get the largest edge size
    g_len = xgi.max_edge_order(H0.H)
    # get the maximum step size for the process
    g_edges = H0.H.num_edges
    ETA, BETA, GAMMA, time_takes, T= experiment(H0, g_len, 200000, batch_size = 10)
    
    pars = {"dataset": data_name, "time": time_takes, "converge": T, "eta": ETA,  "beta": BETA,  "gamma": GAMMA}
    
    with open('sem_results/res_CS2_{}.pkl'.format(data_name), 'wb') as fp:
        pickle.dump(pars, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

def run_bench_mark(data_name):
    
    Dir = r"../data//"
    with open(os.path.join(Dir, data_name + '_pos.pkl'), 'rb') as h: pos_edges = pickle.load(h)
        
    H0 = xgi.Hypergraph(pos_edges)
    H0 = m.GrowingHypergraph(H0)
    ETA, BETA, GAMMA = experiment(H0)
    
    pars = {"dataset": data_name, "eta": np.mean(ETA),  "beta": np.mean(BETA),  "gamma": np.mean(GAMMA, axis = 0)}
    
    with open('sem_results_new/res_{}.json'.format(data_name + "_SEM_new"), 'w') as fp:
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
