import xgi
import numpy as np
import math
import scipy.special as ss
import likelihoods
# from . import likelihoods

class EM:
    
    def __init__(self, H):
        """
        Start with the non-gauranteed version and move on. 
        """
        self.ll_history = []
        self.pars = {"eta" : 0.5, "beta" : 0.5, "gamma" : 0.5}
        
        self.H = H
        
        self.edge_sample_likelihood = likelihoods.edge_sample_likelihood
        
        self.novel_nodes_likelihood = likelihoods.novel_nodes_likelihood
        
        self.nodes_from_hypergraph_likelihood = likelihoods.nodes_from_hypergraph_likelihood
        
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
    
    def fit(self, pars = None, max_iter = 100, tol = 1e-6):
        """
        main loop: initializes parameters and arrays encoding 
        topology, then alternates E and M steps until convergence
        """
        
        # initial guess, if not given then start with a default guess.
        if pars:
            self.pars = pars
        else:   
            self.pars = {"eta" : 0.1, "beta" : 0.1, "gamma" : 0.7}

        # initialize for main loop
        self.ll_history.append(self.marginal_log_likelihood())
        done = False
        i = 0
        
        # do EM until convergence
        while not done: 
            self.E_step()
            self.M_step()
            ll = self.marginal_log_likelihood()
            self.ll_history.append(ll)
            print(f"Step {i}: marginal ll = {ll}")
            if (self.ll_history[-1] - self.ll_history[-2] < tol) or i > max_iter:
                done = True
            i += 1
       
    def marginal_log_likelihood(self):
        """
        at the moment, only implemented to be called within the EM algorithm
        
        this has good vibes but may not be all the way correct
        """
        return np.log(np.nansum(self.likelihood_array*self.intersection_array))
        
    def E_step(self):
        
        k_max = self.edge_sizes.max()
        k = np.arange(k_max + 1)

        # I is the size of e
        # J is the size of f
        # K is the size of e \cap f
        # L is the number of additional novel nodes that 
        # must be added 
        
        I, J, K, L = np.meshgrid(k, k, k, k, indexing = "ij")
        
        
        s1 = self.edge_sample_likelihood(K, J, eta = self.pars["eta"])
        s2 = self.novel_nodes_likelihood(L, beta = self.pars["beta"])
        s3 = self.nodes_from_hypergraph_likelihood(I - K - L, gamma = self.pars["gamma"])

        P = s1*s2*s3
        
        # first index is edge sizes, hopefully
        # probably want to pre-initialize and re-allocate this array 
        # for later 
        self.likelihood_array[:,:,:,:] = P[self.edge_sizes,:,:,:]
        CHI = self.likelihood_array*self.intersection_array
        
        CHI = CHI / CHI.sum(axis = (1, 2, 3))[:, None, None, None]
        
        self.CHI = CHI
    
    def M_step(self):
        k_max = self.edge_sizes.max()
        k = np.arange(k_max + 1)

        # I is the size of e
        # J is the size of f
        # K is the size of e \cap f
        # L is the number of additional novel nodes that 
        # must be added 
        
        I, J, K, L = np.meshgrid(k, k, k, k, indexing = "ij")
        
        # these updates have good vibes but are probably not fully correct yet
        
        
        self.pars["eta"] = np.nansum(self.CHI*(K[self.edge_sizes,:,:,:]-1)) / np.nansum(self.CHI*(self.edge_sizes[:,None, None, None] - K[self.edge_sizes,:,:,:]))
        
        self.pars["beta"] = np.nanmean(np.sum(self.CHI*L[self.edge_sizes,:,:,:], (1, 2, 3)))
        
        self.pars["gamma"] = np.nanmean(np.sum(self.CHI*((I-K-L)[self.edge_sizes,:,:,:]), (1, 2, 3)))
        
        
        
        """
        should turn out to something like (after implementing these methods): 
        
        self.pars["eta"] = self.edge_sample_likelihood.m_step(self.intersection_array, self.CHI)
        
        self.pars["beta"] = self.novel_nodes_likelihood.m_step(self.intersection_array, self.CHI)
        
        self.pars["gamma"] = self.nodes_from_hypergraph_likelihood.m_step(self.intersection_array, self.CHI)   
        """

# likelihoods
# maybe not the nicest API yet, we'll see
        
# def novel_nodes_pmf(x, beta):
#     P = (beta**x)*np.exp(beta)/(ss.factorial(x))
#     P[x < 0] = 0
#     return P
   
# def guaranteed_edge_sample_pmf(x, t, eta):
#     M = (eta**(x-1))*((1-eta)**(t-x)) # ?
#     M[x == 0] = 0
#     M[t < x] = 0
#     return M

# def nodes_from_hypergraph_pmf(x, gamma):
#     M = (gamma**x)*np.exp(-gamma)/(ss.factorial(x))
#     M[x < 0] = 0
#     return M