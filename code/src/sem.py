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
        
        
        # Note very importantly, this new neighborhood de could not be the same as e, 
        # Because if so then there's a high possibility that the only edge overlapping with it is itself
        # When situation like this happens you enter an eternal loop and you fail. 
        # This happened for Math-geology possibly due to the fact that somebody publish paper by themselves, which is natural
        
        
        
        while True: 
            eid = np.random.randint(min_time, self.H.num_edges)
            e   = self.H.edges.members(eid)
            
            # set of all edges that overlap e and arrived before e
            de  = self.H.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
            if len(de) > 0 : 
                break
                
        #print(e, de)
        
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
            p1 = (self.pars["eta"]**(k - 1.0))*((1.0 - self.pars["eta"])**(j - k))
            
            # addition of novel nodes
            p2 = poisson(l, self.pars["beta"])
            
            # addition of nodes from the remainder of the hypergraph (approximation when the number of nodes is large)
            p3 = poisson(i - k - l, self.pars["gamma"])/(ss.binom(self.new_node_sequence[eid-1], i - k - l))
            
            # add to s
            p  = p1*p2*p3
            S += p*s
            P += p
            
            #print(np.array([i, j, k]), e, e_)
        
        
        # dividing by P means that S is the *conditional* distribution over e_ given e, which is what we want. 
        
        #print("old S")
        #print(S)
        
        S /= P
        
        #print("S")
        #print(S)
        #print("P")
        #print(P)
        
        return S
        
    def SEM_step(self, rho, **kwargs):
        """
        this is the function in which we could implement variance-reducing stochastic EM as in that paper. 
        """
        
        # Xie: One of the main thing we could focus on revising here is the step size rho
        # according to the original paper and some research, the convergence of SEM is largely dependent on rho.
        
        
        
        # technically, this is the stochastic E step
        S = self.SE_step(**kwargs)
        
        # this is the stochastic M step
        # technically it's an optimization problem, but we can again do it in closed form
        if (S[0] + S[1]-1) !=0 : 
            eta_update = (S[0]-1)/(S[0] + S[1] - 1)
        else:
            #print("here")
            eta_update = self.pars["eta"]
        
        gamma_update = S[2]
        beta_update = S[3] 
        
        
        
        
        # now we average the current estimates with the new ones
        self.pars["eta"]   = (1-rho)*self.pars["eta"]   + rho*eta_update
        self.pars["gamma"] = (1-rho)*self.pars["gamma"] + rho*gamma_update
        self.pars["beta"]  = (1-rho)*self.pars["beta"]  + rho*beta_update
        
        
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
            

            p1 = (pars["eta"]**(k - 1.0))*((1.0 - pars["eta"])**(j - k))
            
            # addition of novel nodes
            p2 = poisson(l, pars["beta"])
            
            # addition of nodes from the remainder of the hypergraph (approximation when the number of nodes is large)
            p3 = poisson(i - k - l, pars["gamma"])/(ss.binom(self.new_node_sequence[-1], i - k - l))
            
            p  = p1*p2*p3
            P += p
            
            count +=1
        
        proba = P/count
        
        return proba 

def poisson(x, lam):
    return np.exp(-lam)*(lam**(x))/ss.factorial(x)
