from collections import Counter
import random
import numpy as np

class GrowingHypergraph:
    
    def __init__(self, E = None, N = None):
        
        # if user supplies E as a Counter of tuples
        if E: 
            self.E = E
            self.N = set([i for i in e for e in E])
        
        if N: 
            self.N = set(N)
            self.E = []
        
        # otherwise initialize empty
        else:
            self.E = []
            self.N = set()
        
        self.n = len(self.N)
        self.m = len(self.E)
        
    def add_edge(self, e):
        self.E.append(tuple(sorted(e)))
        self.N  = self.N.union(set(e))
        self.n  = len(self.N)
        self.m += 1
    
    def sample_edge(self, eta, gamma):
        
        # randomly select edge from self.E
        # requires Python >= 3.7 to ensure that keys and values 
        # are returned in corresponding order
        
        # conjecture: actually faster to just have a list of edges
        e = self.E[np.random.randint(0, self.m)]
        e_ = []
        
        for node in e: 
            if random.uniform(0,1) < eta:
                e_.append(node)
        
        # ball-dropping implementation to add nodes
        num_left   = len(self.N) - len(e)
        num_to_add = np.random.binomial(num_left, gamma)
        
        # rejection sample
        num_added = 0
        while num_added < num_to_add: 
            candidate = np.random.randint(0, self.n)
            if candidate not in e:
                e_.append(candidate)
                num_added += 1
        
        # nodes_to_add  = np.random.choice(list(self.N.difference(set(e))), num_to_add)
        # e_ += nodes_to_add
        # for node in self.N.difference(set(e)):
        #     if random.uniform(0,1) < gamma: 
        #         e_.append(node)

        self.add_edge(e_)
        return(e_)
    
    def degree_sequence(self):
        degree_seq = Counter({node : 0 for node in self.N})
        for node in self.N:
            for edge in self.E: 
                if node in edge: 
                    degree_seq[node] += 1
        return degree_seq


