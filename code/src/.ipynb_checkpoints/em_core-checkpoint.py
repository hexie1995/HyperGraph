import xgi
import numpy as np
import math
from collections import Counter

# +
# make an initial guess on eta, gamma, beta (beta an int).
# calculate the E step output
# update eta and gamma. 

def E_step(e, H,  heta, hgamma, hbeta, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):
    num_of_edges = H.num_edges
    pzx = []
    
    for i in range(num_of_edges):
        # load saved information
        N_mu = Nmu_vec[i]
        k = k_vec[i]
        Np  = Np_vec[i]
        k_prime = kp_vec[i]
        b = b_vec[i]
        
        bscore = (hbeta**(b)*np.exp(hbeta))/(math.factorial(b))
        
        score = ((1-heta)**(k))*(heta**(N_mu-k))*(hgamma**(k_prime))*((1-hgamma)**(Np-k_prime))*bscore
        #print(score)
        pzx.append(score)
        
    total = sum(pzx)
    pzx_vec = [x/total for x in pzx]
    return pzx_vec, total

def M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):

    
    heta =  1-(np.dot(k_vec, pzx_vec)/np.dot(Nmu_vec ,pzx_vec))
    hgamma = np.dot(kp_vec, pzx_vec)/np.dot(Np_vec,pzx_vec)

    # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
    hbeta = np.dot(b_vec, pzx_vec)/sum(pzx_vec)
    
    #print(heta, hgamma, hbeta)
    return heta, hgamma, hbeta

def EM_update(e, H, eta, gamma, beta, max_iter):
    
    #intial guess
    heta, hgamma, hbeta = eta, gamma, beta
    
    #intialize error term
    true_error = 1
    
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
        #print(e_i)
        #print(e.intersection(e_i))
        k = N_mu - len(e.intersection(e_i))
        Np  = N - N_mu
        k_prime = len(e.intersection(H.nodes-e_i))
        
        #print(e.intersection(H.nodes-e_i))
        new_added = len(e.difference(H.nodes))
        #print(new_added)
        
        
        k_vec.append(k)
        kp_vec.append(k_prime)
        Np_vec.append(Np)
        Nmu_vec.append(N_mu)
        b_vec.append(new_added)
        
    
    #print("kvec", k_vec)
    #print("kpvec", kp_vec)
    #print("Npvec", Np_vec)
    #print("Nmuvec", Nmu_vec)
    #print("bvec", b_vec)
        
        
            
    count = 0
    while count<max_iter:
        
        pzx_vec, total = E_step(e, H, heta, hgamma, hbeta, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)
        #print(pzx_vec)
        #pzx_vec = [x/num_of_edges for x in pzx_vec]
        heta, hgamma, hbeta = M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)    
        count = count+1
        
    #print(pzx_vec)
    return heta, hgamma, hbeta
