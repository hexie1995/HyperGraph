import xgi
import numpy as np
import math
import scipy.special as ss
# import likelihoods
from . import likelihoods

class SEM:
    
    def __init__(self, H, pars = None):
        
        self.ll_history = []
        
        if not pars: 
            pars = {"eta" : 0.5, "beta" : 0.5, "gamma" : 0.5}
        self.pars = pars
        
        self.H = H
        
        self.new_node_sequence = self.H.new_node_sequence()
        
    def SE_step(self, min_time):
        
        eid = np.random.randint(min_time, self.H.num_edges)
        e   = self.H.edges.members(eid)
        
        de  = self.H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
        while (len(de) == 0):
            eid = np.random.randint(min_time, self.H.num_edges)
            e   = self.H.edges.members(eid)
            de  = self.H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
        
        
        l = len([i for i in e if i > self.new_node_sequence[eid-1]])
        i = len(e)
        
        S = np.zeros(4)
        P = 0 
        
        for e_ in de: 
            j = len(e_)
            k = len(e.intersection(e_))
            
            # vector of sufficient statistics
            s = np.array([k, j - k, i - k - l, l])
            
            # guaranteed model for sampling nodes from e_
            p1 = k/j*(self.pars["eta"]**(k - 1.0))*((1.0 - self.pars["eta"])**(j - k))
            
            # addition of novel nodes
            p2 = poisson(l, self.pars["beta"])
            
            # addition of nodes from the remainder of the hypergraph (approximation when the number of nodes is large)
            p3 = poisson(i - k - l, self.pars["gamma"])/(ss.binom(self.new_node_sequence[eid-1], i - k - l))
            
            p  = p1*p2*p3
            S += p*s
            P += p
            
            # print(np.array([i, j, k]), e, e_)

        # should be expectation of sufficient stats under conditional distribution of e_ given e and current parameters. 
        S /= P
        return S
        # return de
    
    def SEM_step(self, rho, **kwargs):
        
        # technically, this is the stochastic E step
        S = self.SE_step(**kwargs)
        
        # this is the stochastic M step
        self.pars["eta"] = (1-rho)*self.pars["eta"] + rho*(S[0]-1)/(S[0] + S[1] - 1)
        
        self.pars["gamma"] = (1-rho)*self.pars["gamma"] + rho*S[2]
        self.pars["beta"]  = (1-rho)*self.pars["beta"] + rho*S[3]
            
def poisson(x, lam):
    return np.exp(-lam)*(lam**(x))/ss.factorial(x)