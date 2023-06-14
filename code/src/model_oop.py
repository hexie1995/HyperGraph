import xgi
import numpy as np
import copy
import math
import scipy.special as ss
from collections import Counter

class GrowingHypergraph():
    
    def __init__(self, H = None, track_branching = True, store_graphs = True):
        
        """
        Unclear what we want track_branching for, but it might be interesting for things like community detection or visualization. 
        """
        if H:
            H = xgi.classes.function.convert_labels_to_integers(H)
            self.H = H
        else:
            self.H = xgi.Hypergraph()
        
        if track_branching: 
            self.track_branching = True
            self.branches = dict()
        
        self.store_graphs = store_graphs
        if store_graphs:
            self.e_vecs = []
            self.H_vecs = []
            
    
        self.nodes = self.H.nodes    
        self.num_nodes = self.H.num_nodes
        self.num_edges = self.H.num_edges
        self.edges = self.H.edges
        self._hypergraph = self.H._hypergraph 
        
        self.node_count_list = []
        self.edge_count_list = []
        
        
    def add_edge(self, e, log_branch = True, store_graphs = True):
        if store_graphs:
            H_temp = copy.deepcopy(self.H)
            self.H_vecs.append(H_temp)
            self.node_count_list.append(H_temp.num_nodes)
            self.edge_count_list.append(H_temp.num_edges)
        self.H.add_edge(e)
        if log_branch: 
            self.branches.update({self.H.num_edges -1: -1}) 
        if store_graphs:
            self.e_vecs.append(set(e))
            
    
    def sample_edge(self, eta, gamma, beta, force_one_node = False):
        
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
        
        self.add_edge(e_, log_branch = False)
        
        
    def sample_edge_v1(self, eta, gamma, beta, force_one_node = False):
        
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


