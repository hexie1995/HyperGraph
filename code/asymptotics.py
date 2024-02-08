import importlib 
m = importlib.import_module(".model", "src")
moments = importlib.import_module(".moments", "src")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import xgi
import scipy.special as ss

# setup for synthetic experiments

NUM_RUNS  = 10
NUM_START = 100
NUM_STEPS = 11100

def init_model(H_data, num_init = 100):
    H = m.GrowingHypergraph()
    
    for i in range(num_init):
        H.add_edge(H_data.H.edges.members(i))
    return H

def er_run(H_data, num_init, num_steps, **kwargs):
    H = init_model(H_data, num_init)

    for i in range(num_steps):
        H.sample_edge_alternative("ER", **kwargs)
    
    return H.intersection_profile(randomize = False)

def model_run(H_data, num_init, num_steps, **kwargs):
    H = init_model(H_data, num_init) 
    for i in range(num_steps):
        H.sample_edge(**kwargs)
    
    return H.intersection_profile(randomize = False)

def add_dicts(list_of_dicts, mean = False):
    
    D = list_of_dicts[0]
    for i in range(1, len(list_of_dicts)):
        d = list_of_dicts[i]
        for k in d.keys():
            if k in D.keys():
                D[k] += d[k]
            else: 
                D[k] = d[k]
           
    if mean: 
        for key in D:
            D[key] = (D[key] / len(list_of_dicts))
    return D

def moving_average(a, n=3):
    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# add empirical

dataset = "email-enron"
H_data = xgi.load_xgi_data(dataset)
H_data = m.GrowingHypergraph(H_data)


NUM_STEPS = min(len(H_data.edge_size_sequence() - NUM_START), NUM_STEPS)

D_data = H_data.intersection_profile(randomize = False)

# parameters for synthetic experiments

model_kwargs = {"eta" : 0.88 , "gamma" : 0.12,  "beta" : 0.013, "force_one_node" : True}

theta, obj = moments.method_of_moments_estimator(H_data, max_init = 1000)

model_kwargs = {"eta" :theta[0] , "gamma" : theta[1],  "beta" : theta[2], "force_one_node" : True}

mean_edge_size = 1 + (model_kwargs["gamma"] + model_kwargs["beta"])/(1 - model_kwargs["eta"])

er_kwargs = {
    "expected_from_graph" : mean_edge_size - 1 - model_kwargs["gamma"], 
    "expected_new" : model_kwargs["gamma"]
    }


# perform experiments

model_experiments = [model_run(H_data, NUM_START, NUM_STEPS, **model_kwargs) for i in range(NUM_RUNS)]
D_model = add_dicts(model_experiments, mean = True)



er_experiments = [er_run(H_data, NUM_START, NUM_STEPS, **er_kwargs) for i in range(NUM_RUNS)]
D_er = add_dicts(er_experiments, mean = True)







fig, ax = plt.subplots(1, 3, figsize = (10, 3.5), sharey=True)
max_k = 6
inferno = mpl.colormaps['inferno'].resampled(max_k+1)
buffer = 100

for i, D in enumerate([D_data, D_model, D_er]):

    for k in sorted(list(D.keys())):
        if k <= max_k:
            averaged = moving_average(D[k][buffer:NUM_STEPS], 1000)
            timesteps = np.arange(buffer, len(averaged) + buffer)
            to_plot = averaged / (2*timesteps)
            # timesteps = np.arange(1, len(D[k])+1)
            ax[i].scatter(timesteps, to_plot, label = rf"$k = {k}$", color = inferno(k-1), s = 1)
            
            
            
            if i == 2:
                x  = np.linspace(timesteps[-1]/5, timesteps[-1], 10)
                # c  = to_plot[-1]/((1.0*t_)**( - k))
                c  = to_plot[1000]/((1.0*timesteps[1000])**(-k)) * 2
                y  = c*(x ** (-k))                
                ax[i].plot(x, y, color = inferno(k-1), linestyle = "--")
            
            if i == -1:
                if k == 1: 
                    x  = np.linspace(timesteps[-1]/5, timesteps[-1], 10)
                    # c  = to_plot[-1]/((1.0*t_)**( - k))
                    c  = to_plot[1000]/((1.0*timesteps[1000])**(-k+1)) * 2
                    y  = c*(x ** (-k+1))                
                    ax[i].plot(x, y, color = inferno(k-1), linestyle = "--")
    ax[i].set(xlabel = "Step")
    ax[i].loglog()               
    ax[i].set(title = ["Enron Email", "Proposed Model", "Growing Erdős-Rényi"][i])
                      
ax[0].legend()
ax[0].set(ylabel = "Edge intersection rate (normalized)")
plt.tight_layout()



plt.savefig("code/figures/er-asymptotics.png")