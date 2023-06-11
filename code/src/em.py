import xgi
import numpy as np
import math
import scipy.special as ss
from collections import Counter

# +
class EM():
    
    def __init__(self, mode = "default"):
        """
        Start with the non-gauranteed version and move on. 
        """
        self.mode = mode
        self.H = None
        
        
    def fit(self, H):
        self.e_vec, self.H_vec = H.e_vecs, H.H_vecs
        self.H = H
        
        
    def par(self, parameters = None, max_iter = 100):
    
        #initial guess, if not given then start with a default guess.
        if parameters:
            heta, hgamma, hbeta = parameters
        else:   
            heta, hgamma, hbeta = 0.1, 0.05, 3
        
        
        
        # in the original code this measn there's only three steps, because intial hypergraph is empty and thus ignored
        # now this is to say that there should be at least one hyper edge exists in order for the model to work
        # at this step the ommitnig first Hyper graph is already done
        node_count = self.H.node_count_list[1:] 
        edge_count = self.H.edge_count_list[1:] 
        # each of the column here in this matrix is an element in the k_vec list
        # IX is the Nmu - k matrix
        IX = xgi.intersection_profile(self.H.H).toarray()
        #print(IX)
        IX = np.triu(IX)
        np.fill_diagonal(IX, 0) 
        invk = IX[:-1,1:]
        invk = invk.transpose()

        # If we proprogate this thing and make it an upper triagonal matrix then this is the Nmu
        # this is necessary because the first hypergraph is omitted if the size of its node set is 0
        # this is the NMU MATRIX
        edge_size_vec = self.H.edge_size_sequence()[:-1]
        Nmu = np.tile(edge_size_vec,(len(edge_size_vec),1)).transpose()
        Nmu = np.triu(Nmu)
        NMUVEC = Nmu.transpose()
        # THIS IS NOT NMU, THIS IS THE LENGTH OF E, THE NEWLY ADDED EDGE
        new_edge_size = self.H.edge_size_sequence()[1:]
        NES = np.tile(new_edge_size,(len(new_edge_size),1)).transpose()
        # KVECS
        KVEC = NMUVEC-invk
        # N vecs
        Np = np.triu(node_count-Nmu)
        NPVEC = Np.transpose()
        # bvec
        b =np.array(node_count[1:] + [self.H.H.num_nodes]) - np.array(node_count)
        BVEC = np.tile(b,(len(b),1)).transpose()
        # This is just kprime because kprime is whatever is left of not the new nodes and not the intersection
        # kprime = new_edge_size - entirely_new_edges - overlap_with_original
        other_nodes_mat = (NES - BVEC) - invk
        KPVEC = np.tril(other_nodes_mat)
        NUM_EDGES = edge_count


        count = 0
        while count<max_iter:


            PZXVEC, total_vec, bscore_vec = self.E_step(heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES)
            cor= self.Test_Correct_vec(NUM_EDGES, total_vec, bscore_vec) 
            heta, hgamma, hbeta = self.M_step(PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES) 

            count = count+1
            self.loglikelihood = cor


        return heta, hgamma, hbeta

    def E_step(self, heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):

        s1 = np.power((1-heta), KVEC)
        s2 = np.power(heta, (NMUVEC- KVEC))
        s3 = np.power((1-hgamma), (NPVEC - KPVEC))
        s4 = np.power(hgamma, KPVEC)

        pzx = s1*s2*s3*s4
        bscore_vec = (np.power(hbeta, BVEC)*np.exp(hbeta))/(ss.factorial(BVEC))
    
        pzx_vec= []        
        total_vec=[]

        for i,ne in enumerate(NUM_EDGES):

            total = sum(pzx[i][0:ne])
            total_vec.append(total)
            pzx_vec.append([x/total for x in pzx[i]])

        PZXVEC = np.array(pzx_vec)
        
        return PZXVEC, total_vec, bscore_vec

    def M_step(self, PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):
        
        heta =  1-(sum(np.multiply(KVEC, PZXVEC).sum(1))/sum(np.multiply(NMUVEC, PZXVEC).sum(1)))
        hgamma = sum(np.multiply(KPVEC, PZXVEC).sum(1))/sum(np.multiply(NPVEC, PZXVEC).sum(1))
        hbeta = sum(np.multiply(BVEC, PZXVEC).sum(1))/np.sum(PZXVEC)  

        #print(heta, hgamma, hbeta)
        return heta, hgamma, hbeta


    def Test_Correct_vec(self, NUM_EDGES, total_vec, bscore_vec):

        curr_vec = np.sum(np.log(total_vec) + np.log(bscore_vec[:,0]) -np.log(NUM_EDGES))

        return curr_vec


    def loss(self):

        return self.loglikelihood

    
    
