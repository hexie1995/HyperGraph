import importlib 
m = importlib.import_module(".model", "src")
em = importlib.import_module(".em", "src")
sem = importlib.import_module(".sem", "src") 
ll = importlib.import_module(".likelihoods", "src")
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import xgi
import copy
import random
import statistics
import matplotlib.pyplot as plt
import scipy.special as ss
import json
import pickle
import os

# The XGI dataset except for disgenenet. 
realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "dawn", "diseasome", "email-enron", "email-eu", "hospital-lyon",
                 "hypertext-conference", "invs13", "invs15", "kaggle-whats-cooking", "malawi-village", "ndc-classes",
                 "ndc-substances", "science-gallery", "sfhh-conference","tags-ask-ubuntu", "tags-math-sx" , 
                 "tags-stack-overflow", "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow"]

# +
# Load Benchmarking dataset 

bench_mark = ["iAF1260b", "iJO1366", "uspto"]


# -

def generate_negative_samples(H, num):
    
    
    # 1. sample the edge size unfiorm randomly from the entire set of edges
    # 2. sample nodes based on the degree sequence of the nodes 
    
    neg_edges = []
    
    for i in range(num):
        
        edge_size = random.choice(H.edges.size.asnumpy()) 
        deg_seq = list(H.nodes.degree.asnumpy())
        normalized_deg_seq =  [float(x)/sum(deg_seq) for x in deg_seq]
        neg_edges.append(np.random.choice(list(H.nodes), edge_size, p=normalized_deg_seq))
    
    return neg_edges

def process_real_world_dataset(data_name, percent_seen = 0.8, num_to_draw = 500):

    # 1. There are multiple ways to to do this, but right now, it is assumed that each edge's probability are calculated independently
    # 2. If a dataset has more than 50k edges, we skip to SEM
    # 3. The number of positive edges sampled from each network is settled to be 1000 for now.
    
    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())

    
    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*percent_seen)
    test_timesteps = total_timesteps - train_timesteps


    if test_timesteps > num_to_draw:
        num_sampled = num_to_draw
    else:
        num_sampled = test_timesteps

    
    
    H1 = xgi.Hypergraph()
    for i in range(train_timesteps):
        H1.add_edge(H0.edges.members(i))

    # this is the graphs that we have observed
    H1 = m.GrowingHypergraph(H1)
    
    # now we build a model using the guaranteed version of the sampler 
    # because that's how we built the original hypergraph
    neg_edges = generate_negative_samples(H1.H, num_sampled)
    pos_edges = []
    
    for i in range(train_timesteps, train_timesteps + num_sampled):    
        pos_edges.append(list(H0.edges.members(i)))
        
    to_pred = [list(x) for x in pos_edges] + [list(x) for x in neg_edges]
    true_label = [1]*num_sampled + [0]*num_sampled
    
    return H0, H1, to_pred, true_label

def SEM_link_prediction(data_name):
    
    
    #H0, H1, edges_pred, labels = process_real_world_dataset(data_name)
    
    H0, H1, edges_pred, labels = process_benchmark_dataset(data_name)
    

    with open('results/res_{}.json'.format(data_name + "_SEM")) as f:
        PARS = json.loads(f.read())
    del PARS["dataset"]

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

    return roc_auc

def SEM_prediction_results(data_name):
    
    AUC = []
    default_trials = range(10)

    for _ in default_trials:

        AUC.append(SEM_link_prediction(data_name))

    np.savetxt('results/auc_{}.txt'.format(data_name), AUC)

# +
# To run the real-world networks in XGI parallely for 10 randomized run, call the following function. 

#with Pool(len(realworld_Hgraphs)) as p:
#    print(p.map(SEM_prediction_results, realworld_Hgraphs))

# +
def process_benchmark_dataset(data_name, percent_seen = 0.8, num_to_draw = 500):

    Dir = r"../data//"
    with open(os.path.join(Dir, data_name + '_pos.pkl'), 'rb') as h: pos_edges = pickle.load(h)
    with open(os.path.join(Dir, data_name + '_neg.pkl'), 'rb') as h: neg_edges = pickle.load(h)


    H0 = xgi.Hypergraph(pos_edges)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())


    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*percent_seen)
    test_timesteps = total_timesteps - train_timesteps


    if test_timesteps > num_to_draw:
        num_sampled = num_to_draw
    else:
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
    

# -

for data_ in bench_mark:
    print(data_)
    print(SEM_link_prediction(data_))
