import xgi 
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
import os 
import pandas as pd
from itertools import product
from scipy.special import binom

os.chdir("code")
import importlib 
m = importlib.import_module(".model", "src")
em = importlib.import_module(".em", "src")
sem = importlib.import_module(".sem", "src") 
ll = importlib.import_module(".likelihoods", "src") 
os.chdir("..")






### PREP

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs("code/figures/moments",exist_ok = True)

### Plot Functions

def read_pars(path = "code/figures/parameters.csv"):
    return pd.read_csv(path).set_index("dataset")

pars = read_pars().to_dict(orient = "index")

def prepare_figure(data_name):
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    par = pars[data_name]
    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
    fig.suptitle(fr"{data_name}:$\eta = ${eta}, $\beta = ${beta},$\gamma = ${gamma}")
    return fig, ax

def prepare_data(data_name):
    H = xgi.load_xgi_data(data_name)
    return m.GrowingHypergraph(H)

def degree_histogram(H, par, ax):
    d = H.degree_sequence()

    # probably need to do log binning in here or something. 
    d = d[d>=10]    
    
    hist, bins = np.histogram(d, bins = min(int(len(d)/5), 100))
    p = hist/hist.sum()

    ax.scatter(bins[:-1], p)
    ax.loglog()
    
    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
    
    exponent = - (1 + 1 / eta)
    ax.set(xlabel = "Degree", ylabel = "Density", title = f"Modeled exponent = {exponent:.2f}")
    
    dmax = d.max()
    dmin = d.min()
    
    upper = dmax / 3
    lower = dmax / 30
    
    x = np.linspace(max(100, lower), upper, 10)
    
    y_upper = p.max()
    y = (x ** exponent) 
    
    y = y / y.sum()
    
    ax.plot(x, y, linestyle = "--", color = "black")
    
    

def edge_size_histogram(H, par, ax):
    
    k = H.edge_size_sequence()

    hist, bins = np.histogram(k, bins = min(int(len(k)/10), 30))
    p = hist / hist.sum()
    ax.semilogy()
    ax.scatter(bins[:-1], p)
    ax.set(xlabel = "Edge Size", ylabel = None)
    v = asymptotic_edge_size_distribution(par, k_max = k.max())
    ax.plot(np.arange(1, len(v)+1), v, color = "black")
    ax.set(ylim = (p[p>0].min()/10, None))

    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
    
    analytic_mean_edge_size = 1 + (gamma + beta) / (1 - eta)
    modeled_mean_edge_size  = (v*np.arange(1, len(v)+1)).sum()

    ax.set(title = f"Mean edge size: {k.mean():.2f}\nModeled mean edge size: {modeled_mean_edge_size:.2f}\nAnalytic mean edge size: {analytic_mean_edge_size:.2f}")

def make_fig(data_name):
    fig, ax = prepare_figure(data_name)
    H = prepare_data(data_name)
    par = pars[data_name]

    
    
    degree_histogram(H, par,ax[0])
    edge_size_histogram(H, par,ax[1])

    plt.tight_layout()
    plt.savefig(f"code/figures/moments/{data_name}.png")

def edge_transition_matrix(par, k_max = 30):
    
    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]

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
    
    Pk = ll.poisson(K, beta)
    Pl = ll.poisson(K, gamma)
    
    
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
    v = eigs[1][:,0]    
    v = v/v.sum()
    return v
    




# make_fig("email-enron")
# make_fig("science-gallery")
# make_fig("ndc-substances")
# make_fig("tags-stack-overflow")
# make_fig("email-eu")
make_fig("hypertext-conference")
# make_fig("contact-high-school")
# make_fig("ndc-classes")
# make_fig("kaggle-whats-cooking")

