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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score,roc_auc_score
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
from itertools import product

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
    

# +
# The XGI dataset except for disgenenet. 
realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "dawn", "diseasome", "email-enron", "email-eu", "hospital-lyon",
                 "hypertext-conference", "invs13", "invs15", "kaggle-whats-cooking", "malawi-village", "ndc-classes",
                 "ndc-substances", "science-gallery", "sfhh-conference","tags-ask-ubuntu", "tags-math-sx" , 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]
# Load Benchmarking dataset 

bench_mark = ["iAF1260b", "iJO1366", "uspto"]

nontemporal = ["disgenenet", "hypertext-conference", "invs13", "invs15", 
               "kaggle-whats-cooking", "malawi-village", "science-gallery", "sfhh-conference"]
temporal = list(set(realworld_Hgraphs)-set(nontemporal))


# -

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

def process_real_world_dataset(data_name, percent_seen = 0.2, num_to_draw = 100000, 
                              randomize = False, use_all_test = False):

    # 1. There are multiple ways to to do this, but right now, it is assumed that each edge's probability are calculated independently
    # 2. If a dataset has more than 50k edges, we skip to SEM
    # 3. The number of positive edges sampled from each network is settled to be 1000 for now.
    
    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    g_len = xgi.max_edge_order(H0.H)
    longest = max(H0.edges.size.aslist())

    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*percent_seen)
    test_timesteps = total_timesteps - train_timesteps


    
    if not use_all_test:
        num_sampled = min(num_to_draw, test_timesteps)
    elif use_all_test:
        num_sampled = test_timesteps

    if not randomize:
        H1 = xgi.Hypergraph()
        for i in range(train_timesteps):
            H1.add_edge(H0.edges.members(i))

        # this is the graphs that we have observed
        H1 = m.GrowingHypergraph(H1)


        pos_edges = []
        for i in range(train_timesteps, train_timesteps + num_sampled):    
            pos_edges.append(list(H0.edges.members(i)))
    
    elif randomize:
        
        print("randomizing")
        H1 = xgi.Hypergraph()
        
        total_edges = list(range(total_timesteps))
        train_edges = random.sample(range(total_timesteps), train_timesteps)
        test_edges = list(set(total_edges).difference(train_edges))        
        
        for i in train_edges:
            H1.add_edge(H0.edges.members(i))

        # this is the graphs that we have observed
        H1 = m.GrowingHypergraph(H1)

        pos_edges = []
        for i in range(num_sampled):
            j = test_edges[i]
            pos_edges.append(list(H0.edges.members(j)))
        
    neg_edges = generate_negative_samples(H1.H, num_sampled)        
    
    to_pred = [list(x) for x in pos_edges] + [list(x) for x in neg_edges]
    true_label = [1]*num_sampled + [0]*num_sampled
    
    return H0, H1, to_pred, true_label, g_len

def process_benchmark_dataset(data_name, percent_seen = 0.2, num_to_draw = 100000, 
                              randomize = False, use_all_test = True):

    Dir = r"../data//"
    with open(os.path.join(Dir, data_name + '_pos.pkl'), 'rb') as h: pos_edges = pickle.load(h)
    with open(os.path.join(Dir, data_name + '_neg.pkl'), 'rb') as h: neg_edges = pickle.load(h)


    if randomize:
        shuffle(pos_edges)
        shuffle(neg_edges)
        
        
    H0 = xgi.Hypergraph(pos_edges)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())


    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*percent_seen)
    test_timesteps = total_timesteps - train_timesteps


    if not use_all_test:
        num_sampled = min(num_to_draw, test_timesteps)
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


# +
def SEM_link_prediction(data_name, randomize_key, use_only_train = True):
    
    
    H0, H1, edges_pred, labels, g_len = process_real_world_dataset(data_name, 
                                                            percent_seen = 0.2, 
                                                            num_to_draw = 100000, 
                                                            randomize = randomize_key, 
                                                            use_all_test = False)

#     H0, H1, edges_pred, labels = process_benchmark_dataset(data_name, 
#                                                             percent_seen = 0.5, 
#                                                             num_to_draw = 100000, 
#                                                             randomize = randomize_key, 
#                                                             use_all_test = False)
    
    
    if not use_only_train:
        with open('results/res_{}.json'.format(data_name + "_SEM")) as f:
            PARS = json.loads(f.read())
        del PARS["dataset"]
        
    elif use_only_train:
        ETA, BETA, GAMMA, time_takes, T= experiment(H0, g_len, 200000, batch_size = 30)
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
    
    
    # Default threshold to median of the results because of balanced class. 
    threshold = np.median(PRED)
    y_true = labels.copy()
    y_pred = [1 if x>=threshold else 0 for x in PRED]
    rec_score = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, PRED)
    
    f1_ = f1_score(y_true, y_pred)

    return roc_auc, rec_score, f1_


# -

def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)


def SEM_prediction_results(data_, trial):
    
    roc_auc, rec_score, f1_ = SEM_link_prediction(data_, randomize_key = False, use_only_train = True)
    D = {"AUC": roc_auc, "recall" : rec_score, "f1": f1_}
    print(D)
    Dir = r"lp_100k_temporal/"
    with open(os.path.join(Dir, data_ + '_CS_'+ str(trial)+ '.pkl'), 'wb') as h: pickle.dump(D, h)


# +
# To run the real-world networks in XGI parallely for 10 randomized run, call the following function. 

smaller_Hgraphs = ["congress-bills", "contact-high-school", "contact-primary-school", "dawn", "diseasome", 
                   "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", "invs13", "invs15", 
                   "kaggle-whats-cooking", "malawi-village", "ndc-classes", "ndc-substances", "science-gallery", 
                   "sfhh-conference"]
                                          
bigger_Hgraphs= ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "tags-ask-ubuntu", "tags-math-sx", 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]

trials = list(range(1))
bench_mark = ["iAF1260b", "iJO1366", "uspto"]



to_para = list(product(temporal, trials))
# SEM_prediction_results(0)

# SEM_prediction_results("tags-math-sx", 0)
# To run the whole xgi datasets for 10 independent trials, uncomment the following. 
with Pool(len(to_para)) as p:
    print(p.starmap(SEM_prediction_results, to_para))


