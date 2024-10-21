# +
import xgi 
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
import os 
import pandas as pd
import pickle
from itertools import product
from tqdm import tqdm
from scipy.special import binom
from multiprocessing import Pool
import importlib 
import time
from sod import *
from collections import Counter, defaultdict
from multiprocessing import Pool
from scipy.interpolate import make_interp_spline, BSpline
m = importlib.import_module(".model", "src")
sem = importlib.import_module(".sem", "src") 
em = importlib.import_module(".em", "src")
ll = importlib.import_module(".likelihoods", "src")

def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)
realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history",
                     "diseasome", "kaggle-whats-cooking", "ndc-classes", "ndc-substances",
                     "tags-ask-ubuntu", "tags-math-sx" , "tags-stack-overflow", "threads-ask-ubuntu", 
                     "threads-math-sx", "threads-stack-overflow",
                     "congress-bills", "contact-high-school", "contact-primary-school",
                     "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", 
                     "invs13", "invs15",  "malawi-village", "science-gallery", "sfhh-conference"]


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
    
    
    max_order = 10
    min_size= 2
    
    # Load the real-world hypergraph and count the number of timestamps depending on the 
    # dataset = "email-enron"
    H1 = xgi.load_xgi_data(dataset)
    H1_edges = list(H1.edges)
    GH = m.GrowingHypergraph(H1)
    timesteps = GH.H.num_edges
    
    with open('sem_results/res_{}.pkl'.format(dataset + "_SEM_new"), "rb") as f:
        d = pickle.load(f)

    eta = np.mean(d["eta"][-50:])
    beta = np.mean(d["beta"][-50:], axis=0)   
    gamma = np.mean(d["gamma"][-50:], axis=0)
    
    #print(eta, beta, gamma)
    
    
    e0 = H1.edges.members(H1_edges[0])
    e1 = H1.edges.members(H1_edges[1])
    
    
    # Get the expected number of new edges to be added from the original network
    expected_from_graph = np.mean(H1.edges.size.aslist())
    
    
    OG, OG_r, OG_esd = initialize_graph_dict(e0, e1)
    H, H_r, H_esd = initialize_graph_dict(e0, e1)
    H_PA, H_PA_r, H_PA_esd = initialize_graph_dict(e0, e1)
    H_ER, H_ER_r, H_ER_esd = initialize_graph_dict(e0, e1)
    
    
    ASSORT_T2, CLUSTER, SF, FES, INTERSECTION = initial_property()
    ASSORT_T2_OG, CLUSTER_OG, SF_OG, FES_OG, INTERSECTION_OG = initial_property()
    ASSORT_T2_PA, CLUSTER_PA, SF_PA, FES_PA, INTERSECTION_PA = initial_property()
    ASSORT_T2_ER, CLUSTER_ER, SF_ER, FES_ER, INTERSECTION_ER = initial_property()
    
    for cc in tqdm(range(2, timesteps), desc ="Timesteps"):
    #for cc in range(2, timesteps):
        
    
        # Initialize HCM edge selection by getting e_
        old_nodes = H.nodes
        e_ = H.sample_edge(eta, gamma, beta, True)
        eid = H.num_edges-1
        e_size = len(e_)
        expected_new = len(e_ -set(old_nodes))
        exact_num = len(e_) - expected_new     
        #print(H.edges)
        H_r, H_esd = aggregate_esd(H, H_r, H_esd, eid, e_)
        
        
        # Get the Original Graph update
        
        e_ = H1.edges.members(H1_edges[cc])
        OG.add_edge(e_)
        OG_r, OG_esd = aggregate_esd(OG, OG_r, OG_esd, eid, e_)
        
        
        # Get the ER and PA based on the expected new nodes that should be added from HCM
        # For PA, think of this as using another HCM, and take the exact same sampled edge, but instead of using eta
        # just randmly sample same number of node based on degree distribution.
        # Add poisson based on mean beta. 
        
        #e_ = H_PA.sample_edge_PA(eta, gamma, beta, True)
        #H_PA_r, H_PA_esd = aggregate_esd(H_PA, H_PA_r, H_PA_esd, eid, e_)
        
        
        e_ = H_ER.sample_edge_alternative("ER", False, exact_num, expected_from_graph, expected_new)
        H_ER_r, H_ER_esd = aggregate_esd(H_ER, H_ER_r, H_ER_esd, eid, e_)

        
        if cc%100 == 0:
            
            append_property(H, H_r, ASSORT_T2, CLUSTER, SF, FES, INTERSECTION)
            append_property(OG, OG_r, ASSORT_T2_OG, CLUSTER_OG, SF_OG, FES_OG, INTERSECTION_OG)
            #append_property(H_PA, H_PA_r, ASSORT_T2_PA, CLUSTER_PA, SF_PA, FES_PA, INTERSECTION_PA)
            append_property(H_ER, H_ER_r, ASSORT_T2_ER, CLUSTER_ER, SF_ER, FES_ER, INTERSECTION_ER)
            
            
        if cc%100 == 0:
            with open('simulation_fig36/' + dataset + '_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_r, fp)
            with open('simulation_fig36/' + dataset + '_orig_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(OG_r, fp)
            #with open('simulation_fig35/' + dataset + '_PA_{}.pkl'.format(cc), 'wb') as fp:
            #    pickle.dump(H_PA_r, fp)
            with open('simulation_fig36/' + dataset + '_ER_{}.pkl'.format(cc), 'wb') as fp:
                pickle.dump(H_ER_r, fp)

            
    
    
    finalize_property(dataset, ASSORT_T2, CLUSTER, SF, FES, INTERSECTION)
    finalize_property(dataset + "_OG", ASSORT_T2_OG, CLUSTER_OG, SF_OG, FES_OG, INTERSECTION_OG)
    #finalize_property(dataset + "_PA", ASSORT_T2_PA, CLUSTER_PA, SF_PA, FES_PA, INTERSECTION_PA)
    finalize_property(dataset + "_ER", ASSORT_T2_ER, CLUSTER_ER, SF_ER, FES_ER, INTERSECTION_ER)
            
            
            
            

def initial_property():
    ASSORT_T2 = []
    CLUSTER = []
    SF = []
    FES = []
    INTERSECTION = []
    
    return ASSORT_T2, CLUSTER, SF, FES, INTERSECTION


def append_property(HH, HH_r, ASSORT_T2_H, CLUSTER_H, SF_H, FES_H, INTERSECTION_H):
    
    max_order = 10
    min_size= 2
    ASSORT_T2_H.append(xgi.degree_assortativity(HH.H, kind = "top-2", exact = True))
    CLUSTER_H.append(np.mean(list(xgi.clustering_coefficient(HH.H).values())))
    
    G = xgi.subhypergraph(HH.H, edges=HH.H.edges.filterby("order", max_order, "leq")).copy()
    G.cleanup(singletons=True)
    
    SF_H.append(edit_simpliciality(G, min_size=min_size))
    FES_H.append(face_edit_simpliciality(G, min_size=min_size))
    
    #H_temp = HH_r.copy()
    #INTERSECTION_H.append(H_temp)
    




def finalize_property(save_name, ASSORT_T2, CLUSTER, SF, FES, INTERSECTION):
    
    d = {}
    d["assort_t2"] = ASSORT_T2
    d["cluster"] = CLUSTER
    d["SF"] = SF
    d["FES"] = FES
    #d["INTERSECTION"] = INTERSECTION
    
    with open('fig_backup/res_{}.pkl'.format(save_name), 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)  


def return_values(parameters, timesteps):
    
    
    eta = parameters["eta"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    
    mean_edge_size = 1 + (expectation(beta) + expectation(gamma)) / (1-eta)
    param_for_alternatives = mean_edge_size - expectation(beta) - 1
    
    max_order = 10
    min_size= 2
    
    ASSORT = []
    ASSORT_TB = []
    ASSORT_T2 = []
    CLUSTER = []
    SF = []
    FES = []
    INTERSECTION = []
    
    e0 = set((0,1,2,3))
    e1 = set((1,2,3,4,5))
    
    
    
    H, H_r, H_esd = initialize_graph_dict(e0, e1)
    
    
    for ii in range(timesteps):
        e_ = H.sample_edge(parameters["eta"], parameters["gamma"], parameters["beta"], True)
        eid = H.num_edges-1
        e_size = len(e_)  
        
        H_r, H_esd = aggregate_esd(H, H_r, H_esd, eid, e_)
        
        if ii%100 == 0:
            start = time.time()
            
            ASSORT.append(xgi.degree_assortativity(H.H, exact = True))
            ASSORT_TB.append(xgi.degree_assortativity(H.H, kind = "top-bottom", exact = True))
            ASSORT_T2.append(xgi.degree_assortativity(H.H, kind = "top-2", exact = True))
            CLUSTER.append(np.mean(list(xgi.clustering_coefficient(H.H).values())))
            
            G = xgi.subhypergraph(H.H, edges=H.H.edges.filterby("order", max_order, "leq")).copy()
            
            SF.append(edit_simpliciality(G, min_size=min_size))
            FES.append(face_edit_simpliciality(G, min_size=min_size))
            
            H_temp = H_r.copy()
            #print(H_temp)
            INTERSECTION.append(H_temp)
            

    d = {}
    d["assort"] = ASSORT
    d["assort_tb"] = ASSORT_TB
    d["assort_t2"] = ASSORT_T2
    d["cluster"] = CLUSTER
    d["SF"] = SF
    d["FES"] = FES
    d["INTERSECTION"] = INTERSECTION

    save_name = "eta_" + str(eta) + "_beta_" + str(round(expectation(beta),2))  + "_gamma_" + str(round(expectation(gamma),2)) 
    
    with open('fig_backup/res_{}.pkl'.format(save_name), 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)  
    
    return 

# +


### Calculate Real-world network properties with the re-built graphs ###

create_HG_rijk("email-enron")
