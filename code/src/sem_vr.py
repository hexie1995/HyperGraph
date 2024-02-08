import xgi
import numpy as np
import math
import scipy.special as ss
# import likelihoods
from . import likelihoods

class SEM_vr:
    
    def __init__(self, H, pars = None):
        
        self.ll_history = []
        
        if not pars: 
            pars = {"eta" : 0.5, "beta" : 0.5, "gamma" : 0.5}
            # we add the average values from the iterations
            self.ETA = [0.5]
            self.BETA = [0.5]
            self.GAMMA = [0.5]
            
        self.pars = pars
        self.H = H

        self.ETA = [pars["eta"]]
        self.BETA = [pars["beta"]]
        self.GAMMA = [pars["gamma"]]
        
        # used to count the iterations this item has go through, update the meanvalue every 100 times.
        self.inner_count = 0
        
    
        # needed to keep track of which nodes were present in the hypergraph 
        # at the time a given edge was formed. 
        self.new_node_sequence = self.H.new_node_sequence()
        
    def SE_step(self, min_time = 0):
        """
        this function is a great place for performance improvements: better use of numpy, more efficient hypergraph queries, memoization, etc. 
        """
        

        # pick a random edge of H
        # it is required that this random edge overlap with some edges
        # that were previous to it
        # otherwise, we assume that the edge was a "root" edge and irrelevant for inference
        while True: 
            eid = np.random.randint(min_time, self.H.num_edges)
            e   = self.H.edges.members(eid)
            
            # set of all edges that overlap e and arrived before e
            de  = self.H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
            if len(de) > 0: 
                break
        
        # now we need to compute the expected sufficient statistics, taking an expectation across all the edges e_ that 
        # overlap e and arrived before e. 
        
        # Let:
        # - k be the size of the intersection e and e_
        # - i be the size of e
        # - j be the size of e_
        # - l be the number of novel nodes in e
        
        # Then, the sufficient statistics are: 
        # - The intersection size k
        # - The number of nodes in e_ not added to e (j - k)
        # - The number of novel nodes l
        # - The number of existing nodes added from the hypergraph, which is i - k - l. 
        
        
        # these two can be calculated without looping over de
        l = len([i for i in e if i > self.new_node_sequence[eid-1]])
        i = len(e)
        
        # vector of sufficient statistics
        S = np.zeros(4)
        
        # this is going to be the marginal likelihood of e
        P = 0 
        
        # loop over de to compute the expectations we need
        for e_ in de: 
            
            # properties of e_ 
            j = len(e_)
            k = len(e.intersection(e_))
            
            # vector of sufficient statistics
            s = np.array([k, j - k, i - k - l, l])
            
            # now we need to weight s by the joint likelihood of e and e_
            
            # guaranteed model for sampling nodes from e_
            p1 = k/j*(self.pars["eta"]**(k - 1.0))*((1.0 - self.pars["eta"])**(j - k))
            
            # addition of novel nodes
            p2 = poisson(l, self.pars["beta"])
            
            # addition of nodes from the remainder of the hypergraph (approximation when the number of nodes is large)
            p3 = poisson(i - k - l, self.pars["gamma"])/(ss.binom(self.new_node_sequence[eid-1], i - k - l))
            
            # add to s
            p  = p1*p2*p3
            S += p*s
            P += p
            
            # print(np.array([i, j, k]), e, e_)

        # dividing by P means that S is the *conditional* distribution over e_ given e, which is what we want. 
        S /= P
        
        return S
        
    def SEM_step(self, rho, **kwargs):
        """
        this is the function in which we could implement variance-reducing stochastic EM as in that paper. 
        """
        
        # technically, this is the stochastic E step
        S = self.SE_step(**kwargs)
        
        # this is the stochastic M step
        # technically it's an optimization problem, but we can again do it in closed form
        eta_update = (S[0] - 1)/(S[0] + S[1] - 1)
        gamma_update = S[2]
        beta_update = S[3]
        
        self.ETA.append(eta_update)
        self.BETA.append(beta_update)
        self.GAMMA.append(gamma_update)
        
        #  this number should change based on what we think a "batch" is. 
        if self.inner_count % 100 ==0 : 
            
            i = self.inner_count
            
            self.initial_eta = eta_update
            self.initial_beta = beta_update
            self.initial_gamma = gamma_update
            
            if self.inner_count == 0:
                self.mean_eta = np.mean(self.ETA[i])
                self.mean_beta = np.mean(self.BETA[i])
                self.mean_gamma = np.mean(self.GAMMA[i])        
            else:
                self.mean_eta = np.mean(self.ETA[i-100:i])
                self.mean_beta = np.mean(self.BETA[i-100:i])
                self.mean_gamma = np.mean(self.GAMMA[i-100:i])
        
        
        # now we average the current estimates with the new ones
        # Method 2(different between 2&3, 2's mean is the overall mean): 
        # self.pars["eta"]   = (1-rho)*self.pars["eta"]   + 0.5*rho*(eta_update + self.mean_eta)
        # Method 1: self.pars["eta"]   = (1-rho)*self.pars["eta"]   + 0.5*rho*(eta_update - initial_eta + self.mean_eta)
        
        
        # Method 3, use only the updated mean in the last 100 iteration, not the previous mean. 
        self.pars["eta"]   = (1-rho)*self.pars["eta"]   + 0.5*rho*(eta_update + self.mean_eta)
        self.pars["gamma"] = (1-rho)*self.pars["gamma"] + 0.5*rho*(gamma_update + self.mean_gamma)
        self.pars["beta"]  = (1-rho)*self.pars["beta"]  + 0.5*rho*(beta_update + self.mean_beta)
        
        self.inner_count += 1

def poisson(x, lam):
    return np.exp(-lam)*(lam**(x))/ss.factorial(x)
