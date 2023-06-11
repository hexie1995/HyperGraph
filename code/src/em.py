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
        
        
    def fit(self, H):
        self.e_vec, self.H_vec = H.e_vecs, H.H_vecs
        
        
    def par(self, parameters = None, max_iter = 100)
    
        #initial guess, if not given then start with a default guess.
        if parameters:
            heta, hgamma, hbeta = parameters
        else:   
            heta, hgamma, hbeta = 0.1, 0.05, 3
        
        H_vec= self.H_vec
        e_vec= self.e_vec
        
        total_edge = H_vec[-1].num_edges
        
        #intialize error term
        true_error = 1

        KVEC = []
        KPVEC = []
        NPVEC = []
        NMUVEC = []
        BVEC = []
        NUM_EDGES = []

        for H,e in zip(H_vec, e_vec):

            #facts about the H graph
            num_of_edges = H.num_edges
            N = H.num_nodes

            #facts about the added edge
            k_vec = []
            kp_vec = []
            Np_vec = []
            Nmu_vec = []
            b_vec = []

            #calculate necessary vectors for EM
            for i in range(num_of_edges):
                e_i = H.edges.members(i)
                N_mu = len(e_i)
                k = N_mu - len(e.intersection(e_i))
                Np  = N - N_mu
                k_prime = len(e.intersection(H.nodes-e_i))
                new_added = len(e.difference(H.nodes))


                k_vec.append(k)
                kp_vec.append(k_prime)
                Np_vec.append(Np)
                Nmu_vec.append(N_mu)
                b_vec.append(new_added)


            if num_of_edges<total_edge:
                for _ in range(total_edge-num_of_edges):
                    k_vec.append(0)
                    kp_vec.append(0)
                    Np_vec.append(0)
                    Nmu_vec.append(0)
                    b_vec.append(new_added)
            # these are all numpy array matrix, list of lists, row = how many edges added, column = how many edges are in each time

            KVEC.append(k_vec)
            KPVEC.append(kp_vec)
            NPVEC.append(Np_vec)
            NMUVEC.append(Nmu_vec)
            BVEC.append(b_vec)

            # this is just a list, their length = how many edges added
            NUM_EDGES.append(num_of_edges)

        #print(KVEC)

        KVEC = np.array(KVEC)
        KPVEC = np.array(KPVEC)
        NPVEC = np.array(NPVEC)
        NMUVEC = np.array(NMUVEC)
        BVEC = np.array(BVEC)
        NUM_EDGES = np.array(NUM_EDGES)


        count = 0
        while count<max_iter:


            PZXVEC, total_vec, bscore_vec = self.E_step(heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES)
            cor= Test_Correct_vec(NUM_EDGES, total_vec, bscore_vec) 
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

        return heta, hgamma, hbeta


    def Test_Correct_vec(self, NUM_EDGES, total_vec, bscore_vec):

        curr_vec = np.sum(np.log(total_vec) + np.log(bscore_vec[:,0]) -np.log(NUM_EDGES))

        return curr_vec


    def loss(self):

        return self.loglikelihood

    
    
