import xgi
import numpy as np
import math
import itertools
import time
from collections import Counter
from multiprocessing import Pool
from model import *
from em_core import *


class GrowingHypergraph:
    
    def __init__(self, H = None):
        if H:
            H = xgi.classes.function.convert_labels_to_integers(H)
            self.H = H
        else:
            self.H = xgi.Hypergraph()
        
    def add_edge(self, e):
        self.H.add_edge(e) 
    
    def sample_edge(self, eta, gamma, beta, force_one_node = False):
        
        e_ = set()
        
        while len(e_) <= 0:
            # first, sample existing edge
            ix = np.random.randint(self.H.num_edges)
            e = self.H.edges.members(ix)
            
            if force_one_node: 
                i = np.random.choice(list(e))
                e_.add(i)
            
            for i in e:
                if np.random.rand() < eta :
                    e_.add(i)
            
            # then, add unrelated nodes from graph
            node_left   =  self.H.nodes-e

            for i in node_left:
                if np.random.rand() < gamma :
                    e_.add(i)
            
            
            # then, add completely novel nodes
            num_to_add = np.random.poisson(beta)
            for i in range(self.H.num_nodes, self.H.num_nodes + num_to_add):
                e_.add(i)
        
        self.add_edge(e_)
        return e_
    
    def degree_sequence(self):
        return self.H.nodes.degree.asnumpy()
    
    def edge_size_sequence(self):
        return self.H.edges.size.asnumpy()
    
    def intersection_profile(self, num_samples = int(1e6)):
        
        edge_sizes = [len(self.H.edges.members(e)) for e in self.H.edges]
        # min_edge_size = min(edge_sizes)
        max_edge_size = max(edge_sizes)
        
        C = np.zeros((max_edge_size+1, max_edge_size+1, max_edge_size+1))
        
        for _ in range(num_samples):
            i, j = 0, 0
            
            while i == j: 
                i = np.random.randint(self.H.num_edges)
                j = np.random.randint(self.H.num_edges)
            
            e = self.H.edges.members(i)
            f = self.H.edges.members(j)
            
            k = len(e.intersection(f))
            
            C[len(e), len(f), k] += 1
            C[len(f), len(e), k] += 1
            
        return C / (2*num_samples)


# +
# make an initial guess on eta, gamma, beta (beta an int).
# calculate the E step output
# update eta and gamma. 

def E_step(e, H,  heta, hgamma, hbeta, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):
    num_of_edges = H.num_edges
    pzx = []
    
    for i in range(num_of_edges):
        # load saved information
        N_mu = Nmu_vec[i]
        k = k_vec[i]
        Np  = Np_vec[i]
        k_prime = kp_vec[i]
        b = b_vec[i]
        
        bscore = (hbeta**(b)*np.exp(hbeta))/(math.factorial(b))
        
        score = ((1-heta)**(k))*(heta**(N_mu-k))*(hgamma**(k_prime))*((1-hgamma)**(Np-k_prime))*bscore
        #print(score)
        pzx.append(score)
        
    total = sum(pzx)
    pzx_vec = [x/total for x in pzx]
    return pzx_vec, total

def M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):

    
    heta =  1-(np.dot(k_vec, pzx_vec)/np.dot(Nmu_vec ,pzx_vec))
    hgamma = np.dot(kp_vec, pzx_vec)/np.dot(Np_vec,pzx_vec)

    # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
    hbeta = np.dot(b_vec, pzx_vec)/sum(pzx_vec)
    
    #print(heta, hgamma, hbeta)
    return heta, hgamma, hbeta

def EM_update(e, H, eta, gamma, beta, max_iter):
    
    #intial guess
    heta, hgamma, hbeta = eta, gamma, beta
    
    #intialize error term
    true_error = 1
    
    #facts about the H graph
    num_of_edges = H.num_edges
    N = H.num_nodes
    
    #facts about the added edge
    k_vec = []
    kp_vec = []
    Np_vec = []
    Nmu_vec = []
    b_vec = []
    
    
    #calculate necessary vectors for EM
    for i in range(num_of_edges):
        e_i = H.edges.members(i)
        N_mu = len(e_i)
        #print(e_i)
        #print(e.intersection(e_i))
        k = N_mu - len(e.intersection(e_i))
        Np  = N - N_mu
        k_prime = len(e.intersection(H.nodes-e_i))
        
        #print(e.intersection(H.nodes-e_i))
        new_added = len(e.difference(H.nodes))
        #print(new_added)
        
        
        k_vec.append(k)
        kp_vec.append(k_prime)
        Np_vec.append(Np)
        Nmu_vec.append(N_mu)
        b_vec.append(new_added)
        
    
    #print("kvec", k_vec)
    #print("kpvec", kp_vec)
    #print("Npvec", Np_vec)
    #print("Nmuvec", Nmu_vec)
    #print("bvec", b_vec)
        
        
            
    count = 0
    while count<max_iter:
        
        pzx_vec, total = E_step(e, H, heta, hgamma, hbeta, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)
        #print(pzx_vec)
        #pzx_vec = [x/num_of_edges for x in pzx_vec]
        heta, hgamma, hbeta = M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)    
        count = count+1
        
    #print(pzx_vec)
    return heta, hgamma, hbeta

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
wrapper_list.append(init_beat)

combs = []

H0 = xgi.random_hypergraph(50, [0.1, 0.01])

for i in itertools.product(*wrapper_list):
    combs.append(i)
    

#eta = 0.6
#gamma = 0.05
#beta = 5
#H0 = xgi.random_hypergraph(50, [0.1, 0.01])
#H1 = GrowingHypergraph(H0)
#e_ = H1.sample_edge(eta, gamma, beta, force_one_node = False)
# -

for count in range(120):
    
    print(count*200, count*200+200)
    count = count+1

    sub_combs = combs[count*200:count*200+200]

    with Pool(len(combs)) as p:
        print(p.map(test_error, combs))
    
    time.sleep(60)


def test_error(parameters):
    
    eta,gamma,beta, ieta,igamma,ibeta = parameters
    
    eta_vec= []
    gamma_vec=[]
    beta_vec=[]

    for _ in range(500):
        H1 = GrowingHypergraph(H0)
        e_ = H1.sample_edge(eta, gamma, beta, force_one_node = False)
        heta, hgamma, hbeta = EM_update(e_, H0, ieta, igamma, ibeta, 100)
        eta_vec.append(heta)
        gamma_vec.append(hgamma)
        beta_vec.append(hbeta)

    with open("output.txt", "a") as f:
        print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta, file=f)
    #print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta)

# +
#eta_vec= []
#gamma_vec=[]
#beta_vec=[]

#ieta =0.9
#igamma = 0.07
#ibeta = 9

#for _ in range(500):
#    H1 = GrowingHypergraph(H0)
#    eta = 0.7
#    gamma = 0.03
#    beta = 5
#    e_ = H1.sample_edge(eta, gamma, beta, force_one_node = False)
#    heta, hgamma, hbeta = EM_update(e_, H0, ieta, igamma, ibeta, 100)
#    eta_vec.append(heta)
#    gamma_vec.append(hgamma)
#    beta_vec.append(hbeta)
    
    
#print(np.mean(eta_vec),np.mean(gamma_vec),np.mean(beta_vec))
# -

errors = np.loadtxt("output.txt")
errors = errors[~np.isnan(errors).any(axis=1)]
errors = np.absolute(errors)

np.mean(errors,axis=0)


