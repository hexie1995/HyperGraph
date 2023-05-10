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
            
        
            
            
            
        
        