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

import importlib 
m = importlib.import_module(".model", "src")
ll = importlib.import_module(".likelihoods", "src") 

# ## PREP

# +
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs("figures/moments",exist_ok = True)


realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history",
                     "diseasome", "kaggle-whats-cooking", "ndc-classes", "ndc-substances",
                     "tags-ask-ubuntu", "tags-math-sx" , "tags-stack-overflow", "threads-ask-ubuntu", 
                     "threads-math-sx", "threads-stack-overflow",
                     "congress-bills", "contact-high-school", "contact-primary-school",
                     "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", 
                     "invs13", "invs15",  "malawi-village", "science-gallery", "sfhh-conference"]


# -

# ## Plot Functions

# +
def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)

def read_pars(path = "figures/parameters.csv"):
    return pd.read_csv(path).set_index("dataset")

def prepare_figure(data_names):
    fig, ax = plt.subplots(3, len(data_names), figsize = (3.3*len(data_names), 9))
    # for i, data_name in enumerate(data_names):
        # par = pars[data_name]
        # eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
        # ax[0,i].set(title = data_name)
        # fig.suptitle(fr"{data_name}:$\eta = ${eta:.3f}, $\beta = ${beta:.3f},$\gamma = ${gamma:.3f}")
    return fig, ax

def prepare_data(data_name, k_max = 50):
    H = xgi.load_xgi_data(data_name)
    k_max = 50
    #k_max = xgi.max_edge_order(H)
    eids = list(H.edges)
    H_ = xgi.Hypergraph()
    for i in range(len(eids)):
        if len(H.edges.members(eids[i])) <= k_max:
            H_.add_edge(H.edges.members(eids[i]))
    
    return m.GrowingHypergraph(H_), k_max

def degree_histogram(H, par, ax, label_y = True):
    d = H.degree_sequence()

    # probably need to do log binning in here or something. 
    d = d[d>=10]    
    
    hist, bins = np.histogram(d, bins = min(int(len(d)/5), 100))
    p = hist/hist.sum()

    ax.scatter(bins[:-1], p,  facecolors='none', edgecolors =  'cornflowerblue', linewidth = 2)
    ax.loglog()
    
    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
    
    exponent = - (1 + 1 / eta)
    ax.set(xlabel = "Degree", xlim = (10, bins.max()*2))
    # ax.set(title = f"Modeled exponent = {exponent:.2f}")
    # ax.text
    
    if label_y:
        ax.set(ylabel = "Density")
    
    dmax = d.max()
    dmin = d.min()
    
    upper = dmax 
    lower = dmax / 5
    
    x = np.linspace(max(100, lower), upper, 10)
    
    y = (x ** exponent) 
    
    y = 3*y / y.sum()
    
    ax.plot(x, y, linestyle = "--", color = "black", zorder = -10, linewidth=1)

    ax.annotate(fr"$\zeta=${exponent:.2f}", xy = (x[0]*2, y[0]/2))
    ax.set(title = "Degree distribution")

def edge_size_histogram(H, par, ax, k_max = 50, show_analytic = False, legend = False):    
    k = H.edge_size_sequence()

    hist, bins = np.histogram(k, bins = min(int(len(k)/10), 50))
    p = hist / hist.sum()
    ax.semilogy()
    ax.scatter(bins[:-1], p, facecolors='none', edgecolors =  "sandybrown", label = "Data", linewidth = 2)
    ax.set(xlabel = "Edge Size", ylabel = None)
    v = asymptotic_edge_size_distribution(par, k_max = k_max)
    # ax.plot(np.arange(1, len(v)+1), v, color = "black", zorder = -10, linestyle = "--", linewidth=1)
    ax.scatter(np.arange(1, len(v)+1), v, c =  "black", zorder = 10, s = 5, label = "Modeled")
    
    ax.set(ylim = (p[p>0].min()/10, None), xlim = (1, k.max() + 2))

    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]
    
    
    
    analytic_mean_edge_size = 1 + (gamma + beta) / (1 - eta)
    modeled_mean_edge_size  = (v*np.arange(1, len(v)+1)).sum()

    title = f"Mean edge size: {k.mean():.2f}\nModeled mean edge size: {modeled_mean_edge_size:.2f}"
    
    if show_analytic: 
        title += f"\nAnalytic mean edge size: {analytic_mean_edge_size:.2f}"
    
    if legend: 
        ax.legend()
    
    ax.set(title = title)


def filter_not_updated(v):
    V = v.copy()
    D = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(D, 1)
    ix = np.argwhere(D == 0)
    dupes = set(np.argwhere(np.isclose(D, 0)).ravel())
    for dupe in dupes: 
        V[dupe] = np.nan 
    return V

def parameter_viz(par, ax, k_max = 50, legend = False, data_name = ""):
    
    # BETA = par["BETA"]
    # GAMMA = par["GAMMA"]
    BETA = filter_not_updated(par["BETA"])
    GAMMA = filter_not_updated(par["GAMMA"])
    
    ax.scatter(np.arange(len(BETA)), BETA, label = r"$\beta$ (novel)", facecolors='none', edgecolors = '#DB7093', linewidth = 2)
    ax.scatter(np.arange(len(GAMMA)), GAMMA, facecolors='none', edgecolors = '#6B8E23', label = r"$\gamma$ (from hypergraph)", marker = "s", linewidth = 2)
    ax.semilogy()
    if legend:
        ax.legend()
    
    title = fr"{data_name}: $\eta = {par['eta']:.2f}$"
    ax.set(xlabel = "Number of added nodes", ylabel = "Probability", title = title)
    


def make_fig(data_names, fname = "multi", show_analytic = False):
    pars = {}
    for data_name in data_names:
    #for data_name in ["email-enron"]:
        pars[data_name] = {}
        with open('code/sem_results_new/res_{}.pkl'.format(data_name + "_SEM_new"), "rb") as f:
            d = pickle.load(f)

        pars[data_name]["eta"] = np.mean(d["eta"][-50:])
        pars[data_name]["beta"] = expectation(np.mean(d["beta"][-50:], axis=0))   
        pars[data_name]["gamma"] = expectation(np.mean(d["gamma"][-50:], axis=0))
        pars[data_name]["BETA"] = np.mean(d["beta"][-50:], axis=0)
        pars[data_name]["GAMMA"] = np.mean(d["gamma"][-50:], axis=0)
        
        
        try:
            pars[data_name]["gamma_vec"] = np.mean(d["gamma"][-50:], axis=0)[0:50]
        except:
            pars[data_name]["gamma_vec"] = np.mean(d["gamma"][-50:], axis=0)
        try:
            pars[data_name]["beta_vec"] = np.mean(d["beta"][-50:], axis=0)[0:50]
        except:
            pars[data_name]["beta_vec"] = np.mean(d["beta"][-50:], axis=0)
            
            
            
    fig, ax = prepare_figure(data_names)
    
    all_y_max_upper = 0
    all_y_min_upper = 1
    all_y_max_lower = 0
    all_y_min_lower = 1
    
    for i, data_name in enumerate(data_names):
        
        H, k_max = prepare_data(data_name)
        par = pars[data_name]

        parameter_viz(par, ax[0, i], k_max = k_max, legend = i == 0, data_name=data_name)
        
        degree_histogram(H, par,ax[1, i], label_y = (i == 0))
        
        # k_max = 100 if data_name == "congress-bills" else 50
        
        edge_size_histogram(H, par, ax[2, i], k_max = k_max, show_analytic=show_analytic, legend = i == 0)
        
        # y_min_upper, y_max_upper = ax[0,i].get_ylim()
        # y_min_lower, y_max_lower = ax[1,i].get_ylim()
        
        # all_y_min_upper = min(all_y_min_upper, y_min_upper)
        # all_y_min_lower = min(all_y_min_lower, y_min_lower)
        # all_y_max_upper = max(all_y_max_upper, y_max_upper)
        # all_y_max_lower = max(all_y_max_lower, y_max_lower)
        
    # for i in range(len(data_names)):
    #     ax[0,i].set(ylim = (all_y_min_upper, all_y_max_upper))
    #     ax[1,i].set(ylim = (all_y_min_lower, all_y_max_lower))
        
    plt.tight_layout()
    plt.savefig(f"figures/distributions/{fname}.pdf")

def edge_transition_matrix(par, k_max = 50):
    
    eta, beta, gamma = par["eta"], par["beta"], par["gamma"]

    GAMMA = np.zeros(k_max)
    for i in range(par["gamma_vec"].shape[0]):
        GAMMA[i] = par["gamma_vec"][i]

        
    BETA = np.zeros(k_max)
    for i in range(par["beta_vec"].shape[0]):
        BETA[i] = par["beta_vec"][i]
    # = par["gamma_vec"]

    # np.pad(par["gamma"], k_max - len(par["gamma"]))

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


# +
# for data_ in realworld_Hgraphs:
#     try:
#         make_fig(data_)
#     except:
#         print("not finished yet")

make_fig([
            "email-enron",
            # "email-enron",
            "ndc-classes", 
            "coauth-mag-history", 
            "tags-ask-ubuntu"
        ], 
        fname = "multi-distribution-fig", 
        show_analytic=False)

    
# make_fig("email-enron")
# make_fig("science-gallery")

# # make_fig("ndc-substances")
# # make_fig("tags-stack-overflow")
# make_fig("email-eu")
# # make_fig("hypertext-conference")
# # make_fig("contact-high-school")
# # make_fig("ndc-classes")
# # make_fig("invs13")
# # make_fig("invs15")
# make_fig("sfhh-conference")
# # make_fig("contact-primary-school")
# # make_fig("diseasome")
# # make_fig("coauth-mag-geology")
# # make_fig("tags-ask-ubuntu")
# #make_fig("coauth-mag-history")

# #make_fig("kaggle-whats-cooking")
# # with Pool(len(realworld_Hgraphs)) as p:
# #      print(p.map(make_fig, realworld_Hgraphs))

