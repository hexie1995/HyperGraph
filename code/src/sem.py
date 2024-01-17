import xgi
import numpy as np
import math
import scipy.special as ss
import functools


class SEM:
    
    def __init__(self, H, edge_sample_likelihood, nodes_from_hypergraph_likelihood, novel_nodes_likelihood, pars = None, memoize = False):

        self.ll_history = []
        
        if not pars: 
            pars = {"eta" : 0.5, "beta" : 0.5, "gamma" : 0.5}
        self.pars = pars
        
        self.H = H
        
        # needed to keep track of which nodes were present in the hypergraph 
        # at the time a given edge was formed. 
        self.new_node_sequence = self.H.new_node_sequence()
        
        self.esl_base = edge_sample_likelihood
        self.nfl_base = nodes_from_hypergraph_likelihood
        self.nnl_base = novel_nodes_likelihood
    
        self.memoize = memoize
        
        self.cache_likelihoods()
        
    def cache_likelihoods(self): 
        
        if self.memoize: 
            self.esl = functools.cache(lambda k, j: self.esl_base(k, j, eta = self.pars["eta"]))
            self.nfl = functools.cache(lambda h: self.nfl_base(h, gamma = self.pars["gamma"]))
            self.nnl = functools.cache(lambda l: self.nnl_base(l, beta = self.pars["beta"]))
            
        else: 
            self.esl = lambda k, j: self.esl_base(k, j, eta = self.pars["eta"])
            self.nfl = lambda h: self.nfl_base(h, gamma = self.pars["gamma"])
            self.nnl = lambda l: self.nnl_base(l, beta = self.pars["beta"])
        
    def sample_edge_with_neighborhood(self, min_time = 0, require_neighbors = True):
        while True: 
            eid = np.random.randint(min_time, self.H.num_edges)
            e   = self.H.edges.members(eid)
            
            # set of all edges that overlap e and arrived before e
            de  = self.H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
            
            if not require_neighbors:
                break
            else:
                if len(de) > 0 : 
                    break
        
        return e, eid, de 
    
    def expected_sufficient_statistics(self, e, eid, de):
                      
        S = np.zeros(4) 
        P = 0 
        
        l = len([i for i in e if i > self.new_node_sequence[eid-1]])
        i = len(e)
        
        for e_ in de: 
            j = len(e_)
            k = len(e.intersection(e_))
            
            # vector of sufficient statistics
            s = np.array([k, j - k, i - k - l, l])

            # compute the likelihood of the vector of sufficient statistics
            p1 = self.esl(k, j)
            
            p2 = self.nfl(i - k - l)
            p3 = self.nnl(l)

            # this is sus -- I think actually this should be the likelihood in batch EM as well. 
            p2 /= (ss.binom(self.new_node_sequence[eid-1], i - k - l))  
            
            p  = p1*p2*p3
            S += p*s
            P += p
            
        S /= P
        
        return S
    
    def SE_step(self, min_time = 0, batch_size = 1):
        """
        this function is a great place for performance improvements: better use of numpy, more efficient hypergraph queries, memoization, etc. 
        """
        
        mean_S = np.zeros(4)    
        for _ in range(batch_size):
            e, eid, de = self.sample_edge_with_neighborhood(min_time, True)
            S          = self.expected_sufficient_statistics(e, eid, de)
        
            mean_S += S
        
        mean_S /= batch_size
        return mean_S

    def SEM_step(self, rho,  **kwargs):
        """
        this is the function in which we could implement variance-reducing stochastic EM as in that paper. 
        """
        
        # Xie: One of the main thing we could focus on revising here is the step size rho
        # according to the original paper and some research, the convergence of SEM is largely dependent on rho.
        
        
        
        # technically, this is the stochastic E step
        S = self.SE_step(**kwargs)
        
        # this is the stochastic M step
        # technically it's an optimization problem, but we can again do it in closed form
        eta_update = (S[0] - 1)/(S[0] + S[1] - 1)
        gamma_update = S[2]
        beta_update  = S[3] 
        
        # now we average the current estimates with the new ones
        self.pars["eta"]   = (1 - rho)*self.pars["eta"]   + rho *eta_update
        self.pars["gamma"] = (1 - rho)*self.pars["gamma"] + rho*gamma_update
        self.pars["beta"]  = (1 - rho)*self.pars["beta"]  + rho*beta_update
        
        self.cache_likelihoods()
        
    def predict(self, H0, e, pars, min_time = 0):
        """
        this function is a great place for performance improvements: better use of numpy, more efficient hypergraph queries, memoization, etc. 
        """
        

        # Instead of a random edge, now we have a definitive edge that will surely have some overlap with its neighbors.
        # We thus directly compute its probability of existence given the parameters of the fit. 
        # set of all edges that overlap e and they definitely arrive before e, so no worries. 
        e = set(e)

        #de =  H0.edge_neighborhood(eid, prior_only = True, as_node_sets = True)

    
        neighbor_edges = []
        for i in e: 
            for eid_ in H0.nodes.memberships(i):
                neighbor_edges.append(eid_)
                    
        neighbor_edges = set(neighbor_edges)
        
        de = [H0.edges.members(eid_) for eid_ in neighbor_edges]
        
        
        #print(e, de)
        
        
        # these two can be calculated without looping over de
        l = len([i for i in e if i > self.new_node_sequence[-1]])
        i = len(e)
        
        # vector of sufficient statistics
        #S = np.zeros(4)
        
        # this is going to be the marginal likelihood of e
        P = 0 
        
        count = 0

        
        for e_ in de: 
            
            # properties of e_ 
            j = len(e_)
            k = len(e.intersection(e_))
            

            p1 =  (k/i)*(pars["eta"]**(k - 1.0))*((1.0 - pars["eta"])**(j - k))
            
            # addition of novel nodes
            p2 = poisson(l, pars["beta"])
            
            # addition of nodes from the remainder of the hypergraph (approximation when the number of nodes is large)
            p3 = poisson(i - k - l, pars["gamma"])/(ss.binom(self.new_node_sequence[-1], i - k - l))
            
            p  = p1*p2*p3
            P += p
            
            count +=1
        
        proba = P/count
        
        return proba 


# -

def poisson(x, lam):
    return np.exp(-lam)*(lam**(x))/ss.factorial(x)
