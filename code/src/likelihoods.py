import numpy as np
import scipy.special as ss

############# 
# Likelihoods
############# 

class Likelihood:
    """
    abstract class for a likelihood defining a forward step of the model
    """
    
    def __init__(self, pmf, m_step):
        
        self.pmf = pmf 
        self.m_step = m_step
    
class EdgeSampleLikelihood(Likelihood):
    """
    likelihood describing the probability of realizing a set of nodes sampled from a specified edge in Step 1 of the model generation algorithm
    """
    
    def __call__(self, selected, total, **pars):
        
        return self.pmf(selected, total, **pars) 

    def m_step(self, selected, total, chi):
        
        return self.m_step(selected, total, chi)        
        
class NovelNodesLikelihood(Likelihood):
    """
    Likelihood describing the probability of realizing a set of nodes added to an edge from outside the current hypergraph node set in Step 3 of the model generation algorithm. 
    """
    
    def __call__(self, num_novel_nodes, **pars):
        
        L = self.pmf(num_novel_nodes, **pars)
        return L
    
    def m_step(self, num_novel_nodes, chi):
        return self.m_step(num_novel_nodes, chi)
    
class NodesFromHypergraphLikelihood(Likelihood): 
    """
    Likelihood function describing the probability of realizing a set of nodes added to an edge from the node set of the current hypergraph, excluding the edge sampled in Step 1. Needs to be corrected by num_nodes_to_add ** (-node_counts) if 1/n scaling is used (recommended so that expected number of nodes added in this step is constant as t increases). 
    """
    def __call__(self, num_nodes_to_add, **pars):
        # need to check this
        L = self.pmf(num_nodes_to_add, **pars)
        return L
    
    def m_step(self, num_nodes_to_add, chi):
        return self.m_step(num_nodes_to_add, chi)


# SPECIFIC LIKELIHOOD INSTANTIATIONS

# estimator likelihood definitions
# pmfs are used in E-step
# m_step estimators are the parameter estimators for each distribution
# used in m-step, using the matrix chi formed in the E-step

# edge sampling
def edge_sample_pmf(x, t, eta):
    return (eta**x)*((1-eta)**(t-x))

def edge_sample_m_step(x, t, chi):
    C = np.tril(chi,-1)
    return (x*C).sum() / (t*C).sum()

esl = EdgeSampleLikelihood(
    pmf = edge_sample_pmf, 
    m_step = edge_sample_m_step
    )

# edge sampling: guaranteed version
def guaranteed_edge_sample_pmf(x, t, eta):
    M = (eta**(x-1))*((1-eta)**(t-x)) # ?
    M[x == 0] = 0
    M[t < x] = 0
    return M

def guaranteed_edge_sample_m_step(x, t, chi):
    C = np.tril(chi,-1)
    top = ((x-1)*C).sum()
    bottom = ((t-1)*C).sum()
    return top / bottom

edge_sample_likelihood = EdgeSampleLikelihood(
    pmf = guaranteed_edge_sample_pmf, 
    m_step = guaranteed_edge_sample_m_step
    )

# addition of novel nodes not previously seen in the hypergraph
def novel_nodes_pmf(x, beta):
    P = (beta**x)*np.exp(-beta)/(ss.factorial(x))
    P[x < 0] = 0
    return P

def novel_nodes_m_step(k, chi):
    return k.mean()

novel_nodes_likelihood = NovelNodesLikelihood(pmf = novel_nodes_pmf, m_step = novel_nodes_m_step)

# addition of nodes from rest of hypergraph
def nodes_from_hypergraph_pmf(x, gamma):
    M = (gamma**x)*np.exp(-gamma)/(ss.factorial(x))
    M[x < 0] = 0
    return M

def nodes_from_hypergraph_m_step(x, chi):
    return (np.tril(chi, -1)*x).sum(axis = 1).mean()

nodes_from_hypergraph_likelihood = NodesFromHypergraphLikelihood(pmf = nodes_from_hypergraph_pmf, m_step = nodes_from_hypergraph_m_step)


# TODO

# - need to adjust the M-step estimators
# - need to implement the EM training loop, similar to prior implementation, probably quite fast
# - need to confidently implement marginal log-likelihood and check 
