import xgi
import numpy as np
import math
import scipy.special as ss
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
        
        score = ((1-heta)**(k))*(heta**(N_mu-k))*(hgamma**(k_prime))*((1-hgamma)**(Np-k_prime))
        #print(score)
        pzx.append(score)
        
    total = sum(pzx)
    pzx_vec = [x/total for x in pzx]
    return pzx_vec, total, bscore

def M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec):

    
    heta =  1-(np.dot(k_vec, pzx_vec)/np.dot(Nmu_vec ,pzx_vec))
    hgamma = np.dot(kp_vec, pzx_vec)/np.dot(Np_vec,pzx_vec)

    # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
    hbeta = np.dot(b_vec, pzx_vec)/sum(pzx_vec)
    
    #print(heta, hgamma, hbeta)
    return heta, hgamma, hbeta

def Test_Correct(prev, num_of_edges, total, bscore):
    
    curr = np.log(total) + np.log(bscore) -np.log(num_of_edges)

    return curr>=prev, curr


def EM_update(e, H, eta, gamma, beta, max_iter):
    
    #intial guess
    heta, hgamma, hbeta = eta, gamma, beta
    
    hprev = -np.inf
    
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
        k = N_mu - len(e.intersection(e_i))
        Np  = N - N_mu
        k_prime = len(e.intersection(H.nodes-e_i))
        new_added = len(e.difference(H.nodes))
        
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
        
    correct = []
            
    count = 0
    while count<max_iter:
        
        pzx_vec, total, bscore = E_step(e, H, heta, hgamma, hbeta, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)
        cor, hprev = Test_Correct(hprev, num_of_edges, total, bscore) 
        heta, hgamma, hbeta = M_step(pzx_vec, k_vec, kp_vec, Np_vec, Nmu_vec, b_vec)    
        count = count+1
        correct.append(cor)
        
    #return heta, hgamma, hbeta
    return heta, hgamma, hbeta, correct
# +
# make an initial guess on eta, gamma, beta (beta an int).
# calculate the E step output
# update eta and gamma. 

def EM_update_vec(e_vec, H_vec, eta, gamma, beta, max_iter):
    
    
    # note that e_vec is now a vector with the sequence of added edges over time
    # here H and max_iter stays the same as before, with H being the initialized hypergraph, and max_iter the iteration of update
    # the averaged output of the first iteration will be the guess of the next iteration, so on and so forth.
    # eta, gamma, beta are all vectorized.
    
    
    total_edge = H_vec[-1].num_edges

    heta, hgamma, hbeta = eta, gamma, beta

    #heta_vec = np.full((total_num_new, total_edge), np.mean(heta))
    #hgamma_vec = np.full((total_num_new, total_edge), np.mean(hgamma))
    #hbeta_vec = np.full((total_num_new, total_edge), np.mean(hbeta))

    #print(heta_vec.shape)
    
    hprev = -np.inf

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
    
    #print(BVEC)
    
    ######## Enter the vectorized iteration ########
    
    correct = []
            
    
    count = 0
    while count<max_iter:
        

        PZXVEC, total_vec, bscore_vec = E_step_vec(heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES)
        cor= Test_Correct_vec(NUM_EDGES, total_vec, bscore_vec) 
        heta, hgamma, hbeta = M_step_vec(PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES) 
        
        count = count+1
        correct.append(cor)
        
        
        #print("HETA", heta_vec)
        #print(hgamma_vec)
        #print(hbeta_vec)
        
        
        
        
    #return heta, hgamma, hbeta
    return heta, hgamma, hbeta, correct

def E_step_vec(heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):
      
    s1 = np.power((1-heta), KVEC)
    s2 = np.power(heta, (NMUVEC- KVEC))
    s3 = np.power((1-hgamma), (NPVEC - KPVEC))
    s4 = np.power(hgamma, KPVEC)

    
    #print(s1)
    
    # this step turn hbeta into a matrix that has the same dimension as the others 
    # however this matrix should be the same on each row 
    bscore_vec = (np.power(hbeta, BVEC)*np.exp(hbeta))/(ss.factorial(BVEC))
    
    #print(bscore_vec)
    
    # this step will return a matrxi num_of_added_edge(i.e time steps T) x num_of_edges at the last time step (i.e. H_n.num_edges) 
    # and the upper left corner of this matrix will be all 0 because those are just the missing information from each time slot
    # and that will be okay because the corresponding k is also 0 there
    # dimension of pzx and KVEC shoud be exactly the same.
    
    pzx = s1*s2*s3*s4
    #print(pzx)
    
    pzx_vec= []        
    total_vec=[]
    
    
    # this is a necessary step to calculate for each ROW of pzx the summation and the averaged value. 
    for i,ne in enumerate(NUM_EDGES):

        total = sum(pzx[i][0:ne])
        total_vec.append(total)
        pzx_vec.append([x/total for x in pzx[i]])

    PZXVEC = np.array(pzx_vec)
    
    #print(PZXVEC)
    return PZXVEC, total_vec, bscore_vec

def M_step_vec(PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):

    heta =  1-(sum(np.einsum('ij, ij->i', KVEC, PZXVEC))/sum(np.einsum('ij, ij->i',NMUVEC, PZXVEC)))
    hgamma = sum(np.einsum('ij, ij->i',KPVEC, PZXVEC))/sum(np.einsum('ij, ij->i', NPVEC, PZXVEC))

    # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
    hbeta = sum(np.einsum('ij, ij->i',BVEC, PZXVEC))/np.sum(PZXVEC)
    #print(hbeta)
    
    return heta, hgamma, hbeta


def Test_Correct_vec(NUM_EDGES, total_vec, bscore_vec):
    
    curr_vec = np.sum(np.log(total_vec) + np.log(bscore_vec[:,0]) -np.log(NUM_EDGES))
    
    return curr_vec


