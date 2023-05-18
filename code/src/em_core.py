import xgi
import numpy as np
from collections import Counter

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
                if np.random.rand() < eta:
                    e_.add(i)
            
            # then, add unrelated nodes from graph
            num_left   = self.H.num_nodes - len(e)
            num_to_add = np.random.binomial(num_left, gamma/self.H.num_nodes)
            
            num_added = 0
            
            while num_added < num_to_add: 
                candidate = np.random.randint(0, self.H.num_nodes)
                # candidate = np.random.choice(self.H.nodes, 1)[0]
                if candidate not in e:
                    e_.add(candidate)
                    num_added += 1
            
            # then, add completely novel nodes
            num_to_add = np.random.poisson(beta)
            for i in range(self.H.num_nodes, self.H.num_nodes + num_to_add):
                e_.add(i)
        
        self.add_edge(e_)
    
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

def E_step(e, H,  eta, gamma, beta):
    num_of_edges = H.num_edges
    pzx = []
    
    for i in num_of_edges:
        # load saved information
        N_mu = Nmu_vec[i]
        k = k_vec[i]
        Np  = Np_vec[i]
        k_prime = kp_vec[i]
        
        score = ((1-eta)**k)*(eta**(N_mu-k))*(gamma**(k_prime))*((1-gamma)**(Np-k_prime)) 
        pzx.append(score)
        
    total = sum(pzx)
    pzx_vec = [x/total for x in pzx]
    return pzx_vec, total


# -

def M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):

    heta = 1-np.dot(Nmu_vec, pzx_vec)/(k_vec,pzx_vec)
    hgamma = np.dot(kp_vec, pzx_vec)/np.dot(Np_vec,pzx_vec)
    
    # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
    hbeta = np.dot(b_vec, pzx_vec)/sum(pzx_vec)
    
    return heta, hgamma, hbeta


# +
def EM_update(e, H, eta, gamma, beta, error, max_iter):
    
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
    for i in num_of_edges:
        e_i = H.edges.members(i)
        N_mu = len(e_i)
        k = len(e.intersection(e_i))
        Np  = H.num_nodes - N_mu
        k_prime = len(e.intersection(H.nodes-e_i))
        new_added = len(e - H.nodes)
        
        k_vec.append(k)
        kp_vec.append(k_prime)
        Np_vec.append(Np)
        Nmu_vec.append(N_mu)
        b_vec.append(new_added)
            
            
    count = 0
    while true_error> error or count<max_iter:
        
        pzx_vec, total = E_step(e, H, heta, hgamma, hbeta)
        heta, hgamma, hbeta = M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)    
        true_error = max([heta-eta, hgamma-gamma, hbeta-beta])
        count = count+1
        
    return heta, hgamma, hbeta
    
    
# -


np.random.poisson(10)
