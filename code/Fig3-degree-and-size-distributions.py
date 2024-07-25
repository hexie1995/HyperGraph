import xgi 
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
import os 
import pandas as pd
import pickle
from itertools import product
from scipy.special import binom
from multiprocessing import Pool
import matplotlib

import importlib 
m = importlib.import_module(".model", "src")
ll = importlib.import_module(".likelihoods", "src") 
plt.style.use('seaborn-v0_8-whitegrid')



# synthetic simulation for first column
pars_synthetic = {
    "eta" : 0.5, 
    "gamma" : np.random.rand(3), 
    "beta" : np.random.rand(3)
}

pars_synthetic["gamma"] = pars_synthetic["gamma"] / pars_synthetic["gamma"].sum()
pars_synthetic["beta"] = pars_synthetic["beta"] / pars_synthetic["beta"].sum()


def read_pars(path = "figures/parameters.csv"):
    return pd.read_csv(path).set_index("dataset")

def prepare_pars(data_name):
    pars = {}
    with open('sem_results_new/res_{}.pkl'.format(data_name + "_SEM_new"), "rb") as f:
        d = pickle.load(f)

    pars["eta"] = np.mean(d["eta"][-50:])
    pars["beta"] = np.mean(d["beta"][-50:], axis=0)
    pars["gamma"] = np.mean(d["gamma"][-50:], axis=0)
    
    return pars

def prepare_data(data_name, k_max = 50):
    H = xgi.load_xgi_data(data_name)
    k_max = 50
    #k_max = xgi.max_edge_order(H)
    eids = list(H.edges)
    H_ = xgi.Hypergraph()
    for i in range(len(eids)):
        if len(H.edges.members(eids[i])) <= k_max:
            H_.add_edge(H.edges.members(eids[i]))
    
    return m.GrowingHypergraph(H_)

def expectation(p):    
    return p @ np.arange(p.shape[0])

def degree_histogram(data_name, ax, label_y = True, offset = (1, 3), pars = None, H = None):
    if pars is None:
        pars = prepare_pars(data_name)
    
    if H is None: 
        H = prepare_data(data_name)
    
    d = H.degree_sequence()

    d = d[d>=10]    
    
    hist, bins = np.histogram(d, bins = min(int(len(d)/5), 100))
    p = hist/hist.sum()

    ax.scatter(bins[:-1], p,  facecolors='none', edgecolors =  'cornflowerblue', linewidth = 2)
    ax.loglog()
    
    eta, beta, gamma = pars["eta"], pars["beta"], pars["gamma"]
    
    mu_beta, mu_gamma = expectation(beta), expectation(gamma)
    
    exponent = - 1 - (1 - eta + mu_gamma + mu_beta) / (1 - eta *(1 - mu_gamma - mu_beta))
    
    ax.set(xlabel = "Degree", xlim = (10, bins.max()*2))
    # ax.set(title = f"Modeled exponent = {exponent:.2f}")
    # ax.text
    
    if label_y:
        ax.set(ylabel = "Density")
    
    dmax = d.max()
    dmin = d.min()
    
    upper = dmax 
    lower = dmax / 5
    
    x = np.linspace(max(100, lower), upper, 10)*offset[0]
    
    y = (x ** exponent) 
    
    y = offset[1]*y / y.sum()
    
    ax.plot(x, y, linestyle = "--", color = "black", zorder = -10, linewidth=1)

    ax.annotate(fr"$\zeta=${exponent:.2f}", xy = (x[0]*2, y[0]/2))
    # ax.set(title = "Degree distribution")
    
    
def edge_size_histogram(data_name, ax, k_max = 50, show_analytic = False, legend = False, horizontal_lim = 20, pars = None, H = None, xlim = None, ylim = None):
    
    if pars is None:
        pars = prepare_pars(data_name)
    
    if H is None: 
        H = prepare_data(data_name)
        
    k = H.edge_size_sequence()

    hist, bins = np.histogram(k, bins = np.arange(k.max()+5))
    p = hist / hist.sum()
    ax.semilogy()
    ax.scatter(bins[:-1], p, facecolors='none', edgecolors =  "sandybrown", label = "Data", linewidth = 2)
    ax.set(xlabel = "Edge Size", ylabel = None)
    v = asymptotic_edge_size_distribution(pars, k_max = k_max)
    # ax.scatter(np.arange(1, len(v)+1), v, c =  "black", zorder = 10, s = 5, label = "Modeled")
    ax.plot(np.arange(1, len(v)+1), v, c =  "black", zorder = 10, label = "Modeled", linestyle = "--", linewidth = 1)  
    
    
    
    # ax.set(ylim = (p[p>0].min()/10, None), xlim = (1, horizontal_lim))
    ax.set(xlim = xlim, ylim = ylim)


    
    
    eta, beta, gamma = pars["eta"], pars["beta"], pars["gamma"]
    
    mu_beta, mu_gamma = expectation(beta), expectation(gamma)
    
    analytic_mean_edge_size = 1 + (mu_gamma + mu_beta) / (1 - eta)
    modeled_mean_edge_size  = (v*np.arange(1, len(v)+1)).sum()

    # title = f"Mean edge size: {k.mean():.2f}\nModeled mean edge size: {modeled_mean_edge_size:.2f}"
    title = "Edge-size distribution"
    
    if show_analytic: 
        title += f"\nAnalytic mean edge size: {analytic_mean_edge_size:.2f}"
    
    if legend: 
        ax.legend()
    
    ax.set(title = title)


def edge_transition_matrix(pars, k_max = 50):
    
    eta, beta, gamma = pars["eta"], pars["beta"], pars["gamma"]

    GAMMA = np.zeros(k_max)
    for i in range(gamma.shape[0]):
        GAMMA[i] = gamma[i]

        
    BETA = np.zeros(k_max)
    for i in range(beta.shape[0]):
        BETA[i] = beta[i]
    

    K = np.arange(0, k_max+1)
    X = K[None,:]
    T = K[:, None]
    
    # M is row-stochastic
    # M[j,h] is the probability of getting h nodes from a sampled edge that contained j nodes. 
    
    M = binom(T - 1, X)*(eta ** (X))*((1 - eta)**(T - 1 - X))
    M[X > T] = 0
    # M[:,0] /= 0
    # M[0,:] = 0
    # print(M)
    
    Pk = BETA[0:k_max+1]
    #print("TEST, K", K)
    Pl = GAMMA[0:k_max+1]
    
    # entry S[i,j] should give the probability of realizing an edge of size j from an edge of size i
    
    S = np.zeros((k_max+1, k_max+1))
    
    for i, h, k, l in product(K, K, K, K):
        j = h + k + l # this is the random part
                      # one less than the true 
                      # edge size
        if j + 1 <= k_max:
            S[i, j+1] += M[i, h] * Pk[k] * Pl[l]
    
    return S[1:,1:]

def asymptotic_edge_size_distribution(*args, **kwargs):
    M = edge_transition_matrix(*args, **kwargs)
    eigs = np.linalg.eig(M.T)
    i = np.argmax(np.real(eigs[0]))
    v = eigs[1][:,i]    
    v = np.real(v/v.sum())
    return v
    



## MAIN PLOTTING



# create synthetic hypergraph 
H_synthetic = m.GrowingHypergraph()
H_synthetic.add_edge((0, 1))
H_synthetic.add_edge((2, 3))

timesteps = int(1e6)
for _ in range(timesteps):
    H_synthetic.sample_edge(pars_synthetic["eta"], 
                            pars_synthetic["gamma"], 
                            pars_synthetic["beta"], True)
    
    
# main figure

plt.rcParams["font.family"] = "Lato"

data_set_1 = "ndc-classes"
data_set_2 = "email-enron"
data_set_3 = "tags-ask-ubuntu"

# for edge-size distribution checking
show_analytic = False

fig, axarr = plt.subplots(2, 4, figsize = (12, 5.5))

# degree distributions
degree_histogram("", axarr[0,0], offset = (0.2, .05), label_y = False, H = H_synthetic, pars = pars_synthetic)
degree_histogram(data_set_1, axarr[0,1], offset = (0.4, 0.5), label_y = False)
degree_histogram(data_set_2, axarr[0,2], offset = (1.0, 0.6), label_y = False)
degree_histogram(data_set_3, axarr[0,3], offset = (0.4, 0.7), label_y = False)

# edge-size distributions

edge_size_histogram("", axarr[1,0], k_max = 30, pars = pars_synthetic, H = H_synthetic, xlim = (0.5, 11.5), ylim = (1e-4, 1), show_analytic=show_analytic, legend = True)
edge_size_histogram(data_set_1, axarr[1,1], k_max = 50, xlim = (0.5, 20.5), ylim = (1e-4, 1), show_analytic=show_analytic)
edge_size_histogram(data_set_2, axarr[1,2], k_max = 50, xlim = (0.5, 20.5), ylim = (1e-4, 1), show_analytic=show_analytic)
edge_size_histogram(data_set_3, axarr[1,3], k_max = 50, xlim = (0.5, 5.5), ylim = (1e-2, 1), show_analytic=show_analytic)


axarr[0,0].set(title = "Synthetic HCM\n\nDegree distribution")
axarr[0,1].set(title = f"{data_set_1}\n\nDegree distribution")
axarr[0,2].set(title = f"{data_set_2}\n\nDegree distribution")
axarr[0,3].set(title = f"{data_set_3}\n\nDegree distribution")


axarr[0,0].set(ylabel = "Probability")
axarr[1,0].set(ylabel = "Probability")
axarr[1,0].set_xticks(np.arange(2, 11, 2))


plt.tight_layout()
plt.savefig("fig/degree-and-edge-sizes.png", dpi = 600, bbox_inches = "tight")