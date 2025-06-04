# +
import cProfile
# cProfile.run("process_real_world_dataset('email-enron')", sort="tottime")
import importlib 
m = importlib.import_module(".model", "src")
em = importlib.import_module(".em", "src")
sem = importlib.import_module(".sem", "src") 
ll = importlib.import_module(".likelihoods", "src")
from matplotlib import pyplot as plt
from multiprocessing import Pool
from random import shuffle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score
import numpy as np
import xgi
import copy
#from tqdm import tqdm
import random
import statistics
import matplotlib.pyplot as plt
import scipy.special as ss
import json
import pickle
import os
import time


bench_mark = ["iAF1260b", "iJO1366", "uspto"]

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
        
        step_size = 1/(100*(i+batch_size+1))
        
        if step_size <1e-7:
            step_size = 1e-7
        
        # change the step size at each loop, linear to the steps. 
        EM.SEM_step(step_size*batch_size, batch_size = batch_size) # both the stochastic E and the M steps are in here
        ETA.append(EM.pars["eta"])
        BETA.append(EM.pars["beta"])
        GAMMA.append(EM.pars["gamma"])
        ADF.append(EM.pars["eta"])
        
        if i % 100 == 0 and i > 100:
            #print("--------")
            #print(f"calculating last averaged differences")
            result = np.mean(ADF[-100:])
            curr_diff = (abs(result - current_eta))/(current_eta)
            current_eta = EM.pars["eta"]
            #print(curr_diff)
    
            if  curr_diff <0.001:
                T = i*batch_size
                early_break = True
                #print("Converged after ", T, "steps")
                break
        
    if early_break is None:
        early_break = False
        T = steps
        
    end = time.time()
    time_takes = end - start
    
    
    return ETA, BETA, GAMMA, time_takes, T


# +
def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)


def generate_negative_samples(H, num):
    
    
    # 1. sample the edge size unfiorm randomly from the entire set of edges
    # 2. sample nodes based on the degree sequence of the nodes 
    
    neg_edges = []
    
    deg_seq = list(H.nodes.degree.asnumpy())
    normalized_deg_seq =  [float(x)/sum(deg_seq) for x in deg_seq]
    edge_seq = H.edges.size.asnumpy()
    
    for i in range(num):
        edge_size = random.choice(edge_seq) 
        neg_edges.append(np.random.choice(list(H.nodes), edge_size, p=normalized_deg_seq))
    
    return neg_edges

def process_benchmark_dataset(data_name, percent_seen = 0.2, num_to_draw = 500, 
                              randomize = True, use_all_test = False):

    Dir = r"../data//"
    with open(os.path.join(Dir, data_name + '_pos.pkl'), 'rb') as h: pos_edges = pickle.load(h)
    with open(os.path.join(Dir, data_name + '_neg.pkl'), 'rb') as h: neg_edges = pickle.load(h)


    if randomize:
        shuffle(pos_edges)
        shuffle(neg_edges)
        
        
    H0 = xgi.Hypergraph(pos_edges)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())
    print(longest)

    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*percent_seen)
    test_timesteps = total_timesteps - train_timesteps


    if test_timesteps > num_to_draw and not use_all_test:
        num_sampled = num_to_draw
    elif use_all_test:
        num_sampled = test_timesteps


    H1 = xgi.Hypergraph()
    for i in range(train_timesteps):
        H1.add_edge(H0.edges.members(i))

    # this is the graphs that we have observed
    H1 = m.GrowingHypergraph(H1)


    neg_edges = neg_edges[0:num_sampled]
    pos_edges = []

    for i in range(train_timesteps, train_timesteps + num_sampled):    
        pos_edges.append(list(H0.edges.members(i)))

    to_pred = [list(x) for x in pos_edges] + [list(x) for x in neg_edges]
    true_label = [1]*num_sampled + [0]*num_sampled

    return H0, H1, to_pred, true_label



def SEM_link_prediction_benchmark(data_name):
    
    start = time.time()
    
    H0, H1, edges_pred, labels = process_benchmark_dataset(data_name, 
                                                           percent_seen = 0.2, 
                                                           num_to_draw = 500, 
                                                           randomize = True, 
                                                           use_all_test = True)
    
    

    #g_len = xgi.max_edge_order(H1.H)
    #print(g_len)
####For these datasets set length 5#####
    g_len = 5 
    
    ETA, BETA, GAMMA, time_takes, T= experiment(H0, g_len, 20000, batch_size = 20)
    
    end = time.time() - start
    
    PARS = {"eta": np.mean(ETA),  "beta": (np.mean(BETA, axis = 0)).tolist(),  "gamma": (np.mean(GAMMA, axis = 0)).tolist()}
        

    EM = sem.SEM(H1,
                 ll.guaranteed_edge_sample_likelihood,
                 ll.nodes_from_hypergraph_likelihood,
                 ll.novel_nodes_likelihood,
                 pars = PARS)
    
    PRED = []
    for edge in edges_pred:
        PRED.append(EM.predict(H0, edge, PARS))

    fpr, tpr, _ = roc_curve(labels, PRED)
    roc_auc = auc(fpr, tpr)
    
    # Default threshold to median of the results because of balanced class. 
    threshold = np.median(PRED)
    y_true = labels.copy()
    y_pred = [1 if x>=threshold else 0 for x in PRED]
    
    rec_score = recall_score(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred)

    
    
    return roc_auc, rec_score, f1_, end


# +
for data_name in bench_mark:
    
    AUC = []
    F1 = []
    TIME = []
    
    for _ in range(1):
        roc_auc, rec_score, f1_, end = SEM_link_prediction_benchmark(data_name)
        AUC.append(roc_auc)
        F1.append(f1_)
        TIME.append(end)
    
    print(data_name, np.mean(AUC), np.mean(F1), np.mean(TIME))
    
    
#     H0, H1, edges_pred, labels = process_benchmark_dataset(data_name, 
#                                                            percent_seen = 0.2, 
#                                                            num_to_draw = 500, 
#                                                            randomize = True, 
#                                                            use_all_test = True)
    
    

#     g_len = xgi.max_edge_order(H1.H)
    
#     print(H1.H.nodes.degree.asnumpy())
#     print(g_len)
    
