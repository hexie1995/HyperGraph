# +
import xgi
import numpy as np
import math
import itertools
import scipy.special as ss
import time
from collections import Counter
from multiprocessing import Pool
#from model import *
from model_oop import *



def test_error_graph(parameters):
    
    #p2 = np.array([[0.6, 0.01], [0.01, 0.6]])
    #sizes2 = [25, 25]
    #H2 = xgi.uniform_HSBM(50, 2, p2, sizes2) # links
    
    #p3 = np.array([[0.6, 0.01], [0.01, 0.6]])
    #sizes2 = [25, 25]
    #H2 = xgi.uniform_HSBM(50, 3, p3, sizes3)
    # right now only 2 edges or 3 edges at a time
    # DCSBM
    
    # n number of nodes, m edges size, 
    # xgi.uniform_HSBM(50, 200, [0.]) more flexible
    # xgi.uniform_HPPM() fewer parameters
    # hypergraph planted partition model
    # 2 community
    H0 = xgi.random_hypergraph(50, [0.1, 0.01])
    eta,gamma,beta, ieta,igamma,ibeta = parameters
    
    eta_vec= []
    gamma_vec=[]
    beta_vec=[]
    
    H_temp = H0
    
    # test a hypergraph that 100 edges has been added since the initial state
    # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself
    C = []
    
    try:
        for _ in range(100):
            H1 = GrowingHypergraph(H_temp)
            e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)
            heta, hgamma, hbeta, correct = EM_update(e_, H_temp, ieta, igamma, ibeta, 100)
            eta_vec.append(heta)
            gamma_vec.append(hgamma)
            beta_vec.append(hbeta)
            H_temp = H1.H
            C.append(correct)


        with open("output_graph.txt", "a") as f:
            print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta, file=f)


        with open("correctness.txt", "a") as f:
            print(C, file=f)
            
    except:
        
        with open("error.txt", "a") as f:
            print(parameters, file=f)
            
            
            
            
        
            
            
            
def test_error_graph_vec(parameters):
    
    H0 = xgi.random_hypergraph(50, [0.1, 0.01])
    eta, gamma, beta, ieta,igamma,ibeta = parameters
    
    eta_temp,gamma_temp,beta_temp = ieta,igamma,ibeta
    
    eta_vec= []
    gamma_vec=[]
    beta_vec=[]
    cor_vec = []
    
    H_temp = H0
    
    e_vec = []
    H_vec = []
    # test a hypergraph that 100 edges has been added since the initial state
    # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself
    
    
    for _ in range(200):
        H_vec.append(H_temp)
        H1 = GrowingHypergraph(H_temp)
        e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)        
        H_temp = H1.H
        e_vec.append(e_)
        

    for _ in range(1):
        eta_temp,gamma_temp,beta_temp,cor_temp = EM_update_vec(e_vec, H_vec, eta_temp,gamma_temp,beta_temp, 300)
        eta_vec.append(eta_temp)
        gamma_vec.append(gamma_temp)
        beta_vec.append(beta_temp)
        cor_vec.append(cor_temp)
        
        
    with open("error_vecs.txt", "a") as f:
        print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta, file=f)
    
    #np.save("results/" + "ETA_" + str(parameters) + ".npy", eta_vec)      
    #np.save("results/" + "GAMMA_" + str(parameters) + ".npy", gamma_vec)      
    #np.save("results/" + "BETA_" + str(parameters) + ".npy", beta_vec)      
    np.save("results/" + "COR_" + str(parameters) + ".npy", cor_vec)      

    return cor_vec

test_eta = [x / 10.0 for x in range(1,9)]
test_gamma = [x / 20.0 for x in range(1,6)]
test_beta = [x for x in range(1,5)]

init_eta = [0.25, 0.5, 0.75]
init_gamma = [0.05, 0.1, 0.15]
init_beta = [2,5]


wrapper_list = []

wrapper_list.append(test_eta)
wrapper_list.append(test_gamma)
wrapper_list.append(test_beta)
wrapper_list.append(init_eta)
wrapper_list.append(init_gamma)
wrapper_list.append(init_beta)

combs = []


for i in itertools.product(*wrapper_list):
    combs.append(i)


# +
### test the function of self testing###
QWQ = GrowingHypergraph()
QWQ.Test_error_random_graph(combs[560])
### test the function of stored edges for a graph 





# +
## test the self saved edges
QAQ = xgi.random_hypergraph(50, [0.1, 0.01])
eta,gamma,beta, ieta,igamma,ibeta = combs[5]
QAQ = GrowingHypergraph(QAQ)

for _ in range(100):
    e_ = QAQ.sample_edge_v1(eta, gamma, beta, force_one_node = False)        

    
#print(QAQ.e_vecs)
#print(QAQ.H_vecs)

heta, hgamma, hbeta, correct = QAQ.EM_update_vec(ieta, igamma, ibeta, 100, use_self = True)

print(heta, hgamma, hbeta)


# +
# x1 = np.array([[1,2,3],[3,4,3]])
# x2 = np.array([[1,2,3],[3,0,1]])
# x3 = np.array([[1,2,0],[3,0,1]])

# #np.einsum('ij, ij->i', x1, x2)

# x4 = [1,2,3]
# x5 = [7,8,9,10]

# for i,j in zip(x4,x5):
#     print(i,j)

# x1[:,0] 
    
    

# #np.einsum('ij, ij->i', x1, x2)


# -



# +
# H0 = xgi.random_hypergraph(50, [0.1, 0.01])
# eta,gamma,beta, ieta,igamma,ibeta = 0.25, 0.01, 3, 0.25, 0.01, 3
# eta_vec= []
# gamma_vec=[]
# beta_vec=[]

# H_temp = H0

# # test a hypergraph that 100 edges has been added since the initial state
# # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself

# for _ in range(1):
#     H1 = GrowingHypergraph(H_temp)
#     e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)
#     heta, hgamma, hbeta,correct = EM_update(e_, H_temp, ieta, igamma, ibeta, 50)
#     eta_vec.append(heta)
#     gamma_vec.append(hgamma)
#     beta_vec.append(hbeta)
#     H_temp = H1.H
# correct

# +
# H0 = xgi.random_hypergraph(50, [0.1, 0.01])
# eta, gamma, beta, ieta,igamma,ibeta = combs[3]

# eta_temp,gamma_temp,beta_temp = ieta,igamma,ibeta

# eta_vec= []
# gamma_vec=[]
# beta_vec=[]
# cor_vec = []

# H_temp = H0

# e_vec = []
# H_vec = []
# # test a hypergraph that 100 edges has been added since the initial state
# # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself
# C = []

# for _ in range(100):
    
#     H_vec.append(H_temp)
#     H1 = GrowingHypergraph(H_temp)
#     e_ = H1.sample_edge_v1(eta, gamma, 5, force_one_node = False)        
#     H_temp = H1.H
#     e_vec.append(e_)

# #print(e_vec)

# #for _ in range(100):
# eta_temp,gamma_temp, beta_temp, cor_temp = EM_update_vec(e_vec, H_vec, eta_temp,gamma_temp,3, 500)
# #     eta_vec.append(heta)
# #     gamma_vec.append(hgamma)
# #     beta_vec.append(hbeta)

# print(eta_temp,gamma_temp, beta_temp)
# # print(np.mean(eta_vec))
# # print(np.mean(gamma_vec))
# # print(np.mean(beta_vec))

# start = time.time()
# parameters = combs[7]
# test_error_graph_vec(parameters)

# print(time.time()-start)
# -

corr = test_error_graph_vec(combs[11])
print(len(corr[0]))
corr

for count in range(120):
    
    print(count*200, count*200+200)
    count = count+1

    sub_combs = combs[count*200:count*200+200]

    with Pool(200) as p:
        print(p.map(test_error_graph_vec, sub_combs))
    
    time.sleep(200)

import numpy as np
errors = np.loadtxt("error_vecs.txt")
#errors = errors[~np.isnan(errors).any(axis=1)]
errors = np.absolute(errors)
np.mean(errors,axis=0)

# +

#eta_vec= np.load("results/" + "ETA_" + str(parameters) + ".npy")      
#gamma_vec = np.load("results/" + "GAMMA_" + str(parameters) + ".npy")      
#beta_vec = np.load("results/" + "BETA_" + str(parameters) + ".npy")      
#cor_vec = np.load("results/" + "COR_" + str(parameters) + ".npy")    
# -


