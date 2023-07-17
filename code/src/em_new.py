import xgi
import numpy as np
import math
import scipy.special as ss


class EM:
    
    def __init__(self, H):
        """
        Start with the non-gauranteed version and move on. 
        """
        self.ll_history = []
        self.eta = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        self.H = H
        
        self.initialize_likelihood_array()
        self.form_edge_intersection_array()

    def initialize_likelihood_array(self):
        """
        preallocate an array to hold likelihood calculations
        """
        
        edge_sizes = self.H.edge_size_sequence()
        m = len(edge_sizes)
        k = edge_sizes.max()
        self.likelihood_array = np.zeros((m, k+1, k+1, k+1))
        
        
    def form_edge_intersection_array(self):
        """
        if we could just store the node entry times or something similar 
        in an array without losing too much speed, we could construct
        this array in parallel. Unclear if necessary given sufficiently 
        large computers...
        """
        
        self.edge_sizes = self.H.edge_size_sequence()
        k_max = np.max(self.edge_sizes)
        m = len(self.edge_sizes)
        
        all_nodes = set()

        # edge id of e
        # size of other edge f
        # size of intersection e \cap f
        # number of novel nodes that must be added to the entire hypergraph to make e

        K = np.zeros((m, k_max+1, k_max+1, k_max+1), dtype = int)

        for eid in self.H.edges:
            e = self.H.H.edges.members(eid)
            neighbor_edges = []
            for i in e: 
                neighbor_edges += [eid_ for eid_ in self.H.nodes.memberships(i) if eid_ < eid] 
                for eid_ in set(neighbor_edges): 
                    f = self.H.H.edges.members(eid_)
                    
                    # size of other edge f
                    i = len(f)
                    
                    # number of novel nodes in e
                    j = len(e.difference(f).difference(all_nodes)) 
                    
                    # size of intersection with f
                    k = len(e.intersection(f))
                    
                    K[eid, i, k, j] += 1
                
            all_nodes = all_nodes.union(e)
            
            
        self.intersection_array = K

        self.ix = self.intersection_array > 0
        
    def marginal_log_likelihood(self):
        """
        at the moment, only implemented to be called within the EM algorithm
        
        this has good vibes but may not be all the way correct
        """
        
        return np.nansum(np.log(self.likelihood_array)*self.intersection_array)
        
        
    def E_step(self):
        
        k_max = self.edge_sizes.max()
        k = np.arange(k_max + 1)

        # I is the size of e
        # J is the size of f
        # K is the size of e \cap f
        # L is the number of additional novel nodes that 
        # must be added 
        
        I, J, K, L = np.meshgrid(k, k, k, k, indexing = "ij")
        
        s1 = guaranteed_edge_sample_pmf(K, J, eta = self.eta)
        s2 = novel_nodes_pmf(L, beta = self.beta)
        s3 = nodes_from_hypergraph_pmf(I - K - L, gamma = self.gamma)

        P = s1*s2*s3
        
        # first index is edge sizes, hopefully
        # probably want to pre-initialize and re-allocate this array 
        # for later 
        self.likelihood_array[:,:,:,:] = P[self.edge_sizes,:,:,:]
        CHI = self.likelihood_array*self.intersection_array
        
        CHI = CHI / CHI.sum(axis = (1, 2, 3))[:, None, None, None]
        
        self.CHI = CHI
        
        

        
        
    
    
    
# likelihoods
# maybe not the nicest API yet, we'll see
        
def novel_nodes_pmf(x, beta):
    P = (beta**x)*np.exp(beta)/(ss.factorial(x))
    P[x < 0] = 0
    return P
   
def guaranteed_edge_sample_pmf(x, t, eta):
    M = (eta**(x-1))*((1-eta)**(t-x)) # ?
    M[x == 0] = 0
    M[t < x] = 0
    return M

def nodes_from_hypergraph_pmf(x, gamma):
    M = (gamma**x)*np.exp(-gamma)/(ss.factorial(x))
    M[x < 0] = 0
    return M
