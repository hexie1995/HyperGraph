# +
import json
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgi
from decimal import Decimal
import os
import time
import importlib
import scipy.special as ss
import functools
import xgi
from collections import Counter
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS

m = importlib.import_module(".model", "src")
sem = importlib.import_module(".sem", "src") 
em = importlib.import_module(".em", "src")
ll = importlib.import_module(".likelihoods", "src")


np.random.seed(0)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Lato"


def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)

def KL(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def experiment_KL_CS(parameters, steps, beta_mu, gamma_mu, ax = None, batch_size = 1, verbose = False, legend_on = True):
    
    # KL convergence and changed step size + stopping condition.
    
    eta = parameters["eta"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    
    
    timesteps = int(1e4)

    H = m.GrowingHypergraph()
    H.add_edge((0, 1))
    H.add_edge((2, 3))
    
    for _ in range(timesteps):
        e_ = H.sample_edge(eta, gamma, beta, True)
    #added.append(num)
    
    print(f"This hypergraph has {len(H.edge_size_sequence())} edges and {len(H.degree_sequence())} nodes.")

    
    init_pars = {"eta"   : 0.5, 
             "beta"  : [1/10]*10, 
             "gamma" : [1/10]*10}
    
    if not ax: 
        fig, ax = plt.subplots(1, 1)
        
    EM = sem.SEM(H, 
             ll.guaranteed_edge_sample_likelihood,
             ll.nodes_from_hypergraph_likelihood,
             ll.novel_nodes_likelihood,
             pars = init_pars.copy(), 
             memoize = True)
    
    epochs = int(steps / batch_size)
    
    ETA   = []
    BETA  = []
    GAMMA = []
    ADF =[]
    early_break = None
    

    current_eta = 100
    # main loop
    for i in range(epochs):

        if i % max(int(epochs/20), 1) == 0 and verbose:
            print("--------")
            print(f"Completed epoch {i}")
            print(EM.pars)
            
        EM.SEM_step(1/(i+1)*batch_size, batch_size = batch_size) # both the stochastic E and the M steps are in here
        ETA.append(abs(EM.pars["eta"] - eta))
        BETA.append(KL(EM.pars["beta"], beta))
        GAMMA.append(KL(EM.pars["gamma"], gamma))
        ADF.append(EM.pars["eta"])
        
        
        if i % 200 == 0 and i>1000 :
            print("--------")
            print(f"calculating last averaged differences")
            result = np.mean(ADF[-200:])
            curr_diff = abs(result - current_eta)/current_eta
            current_eta = EM.pars["eta"]
            print(curr_diff)
    
            if  curr_diff <0.01:
                T = i*batch_size
                early_break = True
                print("Converged after ", T, "steps")
                break
        
    if early_break is None:
        early_break = False
        
    # how'd we do? 
    if early_break:
        T = np.arange(i*batch_size+1)
    else:
        T = np.arange(epochs)
    ax.plot(T, ETA,   label = r"$|\hat{\eta} - \eta|$ for $\eta$ = " + str(eta) , color = "cornflowerblue")
    ax.plot(T, BETA,  label = r"$KL(\hat{\beta}, \beta)$ for $\mu_{\beta}$ = " + str(beta_mu), color = "sandybrown")
    ax.plot(T, GAMMA, label = r"$KL(\hat{\gamma}, \gamma)$ for $\mu_{\gamma}$ = " + str(gamma_mu), color = "olivedrab")


#     ax.plot(T, np.ones((epochs,))*eta, color = "blue", linestyle = "dashed")
#     ax.plot(T, np.ones((epochs,))*expectation(beta), color = "black", linestyle = "dashed")
#     ax.plot(T, np.ones((epochs,))*expectation(gamma), color = "grey", linestyle = "dashed")
    
    ax.set_yscale('log')
    ax.set(xlabel = "Algorithmic timestep")
    if legend_on:
        ax.set(ylabel = "Error in parameter estimate")
    ax.legend()
    
    return ETA, BETA, GAMMA


# +
eta_1 = 0.3
eta_2 = 0.7


p_beta_1 = np.random.poisson(lam=2.0, size=10)
p_beta_1 = [x+1 for x in p_beta_1]
msum = sum(p_beta_1)
p_beta_1 = [x/msum for x in p_beta_1]
p_beta_2 = np.random.poisson(lam=3.0, size=10)
p_beta_2 = [x+1 for x in p_beta_2]
msum = sum(p_beta_2)
p_beta_2 = [x/msum for x in p_beta_2]

p_gamma_1 = np.random.poisson(lam=3.0, size=10)
p_gamma_1 = [x+1 for x in p_gamma_1]
msum = sum(p_gamma_1)
p_gamma_1 = [x/msum for x in p_gamma_1]
p_gamma_2 = np.random.poisson(lam=2.0, size=10)
p_gamma_2 = [x+1 for x in p_gamma_2]
msum = sum(p_gamma_2)
p_gamma_2 = [x/msum for x in p_gamma_2]

# +
### Calculate Synthetic Network Properties ###

fig, axes = plt.subplots(1, 2, figsize = (10, 4), sharey= True)
axes = axes.flatten()


PARS = { "eta": eta_1,
        "beta": p_beta_1,
       "gamma": p_gamma_1}

experiment_KL_CS(PARS, 50000, 3, 4, ax = axes[0], batch_size = 1, verbose = False)

PARS = { "eta": eta_2,
        "beta": p_beta_2,
       "gamma": p_gamma_2}

experiment_KL_CS(PARS, 50000, 4, 3, ax = axes[1], batch_size = 1, verbose = False, legend_on = False)

plt.savefig("figures/FigS1.pdf",bbox_inches='tight')
