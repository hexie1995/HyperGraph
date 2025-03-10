import xgi
import numpy as np
from collections import Counter
import random

class GrowingHypergraph:
    
    def __init__(self, H = None, track_branching = True):
        """
        Unclear what we want track_branching for, but it might be interesting for things like community detection or visualization. 
        """
        if H:
            H = xgi.utils.utilities.convert_labels_to_integers(H)
            self.H = H
        else:
            self.H = xgi.Hypergraph()
        
        if track_branching: 
            self.track_branching = True
            self.branches = dict()
        self.nodes = self.H.nodes    
        self.num_nodes = self.H.num_nodes
        self.num_edges = self.H.num_edges
        self.edges = self.H.edges
        self._hypergraph = self.H._hypergraph 
        
        
    def add_edge(self, e, log_branch = True):
        self.H.add_edge(e)
        if log_branch: 
            self.branches.update({self.H.num_edges - 1: -1}) 
        self.num_edges += 1
    
    def sample_edge(self, eta, gamma, beta, force_one_node = False):
        
        ## Note how 123 sizes are returned here. 1: eta, 2: gamma, 3: beta. 
        
        e_ = set()
        
        while len(e_) <= 0:
            # first, sample existing edge
            ix = np.random.randint(self.H.num_edges)
            
            # if we are tracking the branching process
            # log that edge about to be formed was formed from edge ix. 
            if self.track_branching:
                self.branches.update({self.H.num_edges + 1 : ix})
            
            e = self.H.edges.members(ix)
            
            if force_one_node: 
                i = np.random.choice(list(e))
                e_.add(i)
            
            for i in e:
                if np.random.rand() < eta:
                    e_.add(i)
            
            
            # calculate the size1 
            size1 = len(e_)
            
            # DIFFERENCES
            # then, add unrelated nodes from graph
            num_left   = self.H.num_nodes - len(e)
            num_to_add1 = np.random.choice(len(gamma), 1, p = gamma)[0]
            
            num_added = 0
            
            if num_left < num_to_add1:
                num_to_add1 = num_left
            
            
            
            while num_added < num_to_add1 and num_to_add1 != 0: 
                candidate = np.random.randint(0, self.H.num_nodes)
                # candidate = np.random.choice(self.H.nodes, 1)[0]
                if candidate not in e:
                    e_.add(candidate)
                    num_added += 1
            
            # calculate the size2
            size2 = len(e_) - size1
            # then, add completely novel nodes
            
            num_to_add2 = np.random.choice(len(beta), 1, p = beta)[0]
            
            for i in range(self.H.num_nodes, self.H.num_nodes + num_to_add2):
                e_.add(i)
            
            # calculate the size3
            size3 = len(e_) - size2
        
        
        self.add_edge(e_, log_branch = False)
        
        return e_, size1, size2, size3
    
    
    
    
    def sample_edge_alternative(self, method, A, B, expected_new):
        """
        growing Erdos-Renyi graph or preferential attachment graph based on TECH
        
        input:
        method: str, decide whether to use ER or PA, only two options. 
        exact: bool, decides whether the edge size is exactly the same as the TECH model at each timestamp.
        exact_num_nodes: int, see above, but also minus the number of newly added nodes, so be careful here.
        expected_from_graph: int, expected number of edges, only is needed when exact = False. 
        expected_new: int, new nodes added from the completely novel nodes paradigm. 
        
        
        A + B = expected_from_graph
        A = #nodes copied from one of the sampled edge
        B = #nodes copied from the rest of the hgraph
        
        A, B both should be dependent on HCM
        expected_new is also entirely dependent on HCM. 
        
    
        output:
        e_: set, the newly added edge in xgi edge format (a set). 
        self: at the same time updates the hypergraph to add one edge at a time. 
        
        """
        
        e_ = set()
        
        if method == "ER": 
            to_add = np.random.choice(range(self.H.num_nodes), size = A+B, replace = False)
        elif method == "PA":
            d = self.degree_sequence()
            dist = d / d.sum()
            to_add1 = np.random.choice(range(self.H.num_nodes), size = A, replace = False, p = dist) 
            nodes_left = list(set(list(range(self.H.num_nodes))) - set(to_add1))
            to_add2 = np.random.choice(nodes_left, size = B, replace = False)
            to_add = np.concatenate((to_add1, to_add2), axis=None)
            
        for i in to_add:   
            e_.add(i)
        
    
        num_new = np.random.poisson(expected_new)
        for i in range(self.H.num_nodes, self.H.num_nodes + num_new):
            e_.add(i)
        
        self.add_edge(e_, log_branch = False)
        return e_
     
    
    def edge_predecessor(self, edge_ix):
        """
        return the index of the edge used to sample the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        else:
            return self.branches[edge_ix]
    
    def edge_ancestor(self, edge_ix):
        """
        return the root of the branching tree containing the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        
        ix = edge_ix
        while self.branches[ix] != -1:
            ix = self.branches[ix]
        return ix
          
    def edge_lineage(self, edge_ix):
        """
        sequence of edges from the supplied edge index to the root of the branching tree containing the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        
        ix = edge_ix
        lineage = [ix]
        
        while self.branches[ix] != -1:
            ix = self.branches[ix]
            lineage.append(ix)
        
        return lineage 
        
    def degree_sequence(self):
        return self.H.nodes.degree.asnumpy()
    
    def edge_size_sequence(self):
        return self.H.edges.size.asnumpy()
    
    def rw_laplacian(self, **kwargs):
        A = xgi.adjacency_matrix(self.H, **kwargs)
        D = np.diag(A.sum(axis = 1))
        D_inv = np.linalg.inv(D)
        L = D_inv @ (D - A)
        return L
    
    def intersection_array(self, num_samples = int(1e6), normalize = False):
        """
        deprecated, sampling-based, not time-dependent
        """
        
        k_max = self.edge_size_sequence().max()
         
        C = np.zeros((k_max+1, k_max+1, k_max+1))
        
        
        num_edges = self.H.num_edges
        
        random_nums = np.random.randint(0, num_edges, size = (2*num_samples, 2))
                
        current_random_num = 0
        for _ in range(num_samples):
            i, j = 0, 0
            
            while i == j: 
                i = random_nums[current_random_num, 0]
                j = random_nums[current_random_num, 1]
                current_random_num += 1
            
            e = self.H.edges.members(i)
            f = self.H.edges.members(j)
            
            k = len(e.intersection(f))
            
            C[len(e), len(f), k] += 1
        
        C = C + np.transpose(C, (1, 0, 2))
        
        if normalize: 
            C = C / C.sum(axis = 2, keepdims = True)
            
        return C
    
    # def intersection_array(self, normalize = False, sample = None):
        
    #     k_max = self.edge_size_sequence().max()
    #     eids = list(self.H.edges)
    #     m_edges = len(eids)
    #     P = np.zeros((k_max+1, k_max+1, k_max+1))
        
    #     if not sample: 
    #         for eid in eids:
    #             e   = self.H.edges.members(eid)
    #             de = self.edge_neighborhood(eid, as_node_sets = True)
    #             for e_ in de: 
    #                 k = len(e.intersection(e_))
    #                 P[len(e), len(e_), k] += 1

        
    #     if normalize: 
    #         P = P / P.sum(axis = (1, 2), keepdims = True)
    #         # P[np.isnan(P)] = 0
    #     return P 
                
        
    
    def intersection_profile(self, randomize = False, interval = 1):
        
        if randomize:
            GH = self.random_reindexed_hypergraph()
        else: 
            GH = self
        
        d = dict()
        
        eids = list(GH.H.edges)
        m_edges = len(eids)
        
        timesteps = np.arange(1, m_edges // interval)*interval
        
        for i in range(0, m_edges // interval-1):
            eid = eids[timesteps[i]]
            e   = GH.H.edges.members(eid)
            de  = GH.edge_neighborhood(eid, prior_only = True, as_node_sets = True)
            
            for e_ in de: 
                k = len(e.intersection(e_))
                if k not in d: 
                    d[k] = np.zeros((m_edges // interval) - 1)
                d[k][i] += 1
        

        D = {k : np.cumsum(d[k]) / (range(1, m_edges//interval)*(timesteps)) for k in d}
        
        # return d, timesteps
        return D, timesteps

    def random_reindexed_hypergraph(self):
        eids = list(self.H.edges)
        random.shuffle(eids)
        
        H = xgi.Hypergraph()
        for i in range(len(eids)):
            H.add_edge(self.H.edges.members(eids[i]))
        
        return GrowingHypergraph(H, track_branching = False)
        
        

    
    def edge_neighborhood(self, eid, prior_only = False,as_node_sets = False):
        """
        """
        
        
        e = self.H.edges.members(eid)
        set0 = set(e)
        neighbor_edges = []
        
        
        
        for i in e: 
            for eid_ in self.H.nodes.memberships(i):
                if ((not prior_only) or (eid_ <= eid)) and (eid_ != eid): 
                    neighbor_edges.append(eid_)

        
        neighbor_edges = set(neighbor_edges)
        if as_node_sets: 
            return [self.H.edges.members(eid_) for eid_ in neighbor_edges]
        else: 
            return neighbor_edges
    
    def new_node_sequence(self):
        """
        entry t of this list is the id of the most recent node added by time t
        used for novel node calculations and things like that
        """
        n_max = 0
        timesteps = [0 for _ in range(len(self.H.edges))]
        for eid in self.H.edges:
            e = self.H.edges.members(eid)
            i_max = max(e)
            if i_max > n_max: 
                n_max = i_max
            timesteps[eid] = n_max
        return timesteps

        

