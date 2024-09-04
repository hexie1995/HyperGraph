# +
import xgi 
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np
import os 
import pandas as pd
import pickle
import matplotlib.colors as mcolors
from itertools import product
from scipy.special import binom
from multiprocessing import Pool
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import importlib 
import time
from sod import *
from collections import Counter, defaultdict
from multiprocessing import Pool
from scipy.interpolate import make_interp_spline, BSpline
m = importlib.import_module(".model", "src")
sem = importlib.import_module(".sem", "src") 
em = importlib.import_module(".em", "src")
ll = importlib.import_module(".likelihoods", "src")

plt.style.use('seaborn-v0_8-whitegrid')

def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)
realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history",
                     "diseasome", "kaggle-whats-cooking", "ndc-classes", "ndc-substances",
                     "tags-ask-ubuntu", "tags-math-sx" , "tags-stack-overflow", "threads-ask-ubuntu", 
                     "threads-math-sx", "threads-stack-overflow",
                     "congress-bills", "contact-high-school", "contact-primary-school",
                     "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", 
                     "invs13", "invs15",  "malawi-village", "science-gallery", "sfhh-conference"]


def process_intersect(Rmat):
    rk = []
    for r in Rmat:
        
        #print(r)

        i_size = max([x[0] for x in r.keys()])
        j_size = max([x[1] for x in r.keys()])
        k_size = max([x[2] for x in r.keys()])


        #print(i_size, j_size, k_size) 

        r_ijk = np.zeros(shape=(i_size+1, j_size+1, k_size+1))

        for key in r.keys():
            r_ijk[key] = r[key]

        # Transpose the array to bring the first two axes to the end
        array_transposed = np.transpose(r_ijk, (2, 0, 1))
        # Apply np.triu along the last two axes
        upper_triangle_transposed = np.triu(array_transposed)
        # Transpose the result back to the original shape
        upper_triangle = np.transpose(upper_triangle_transposed, (1, 2, 0))


        normalized_rijk = upper_triangle/np.sum(upper_triangle)

        rk.append(np.sum(normalized_rijk, axis = (0,1)))
        
    return rk


properties = [ "Assortativity_Top2", "Clustering Coefficient", "Edit Simpliciality",  "Intersection Sizes"]
idx = ["assort_t2", "cluster", "SF",  "INTERSECTION"]
fig, ax = plt.subplots(1, 4, figsize = (10, 3), sharey= False)
ax = ax.flatten()
inferno = ["sandybrown", "cornflowerblue", "palevioletred", "olivedrab",
          "sandybrown", "cornflowerblue", "palevioletred", "olivedrab"]                                   
                                        
timeline = [x*100 for x in range(1, 101)]

suffix = ["", "_OG", "_PA", "_ER"]
labb = ["HCM", "Orig", "PA", "ER"]

# data_name = "invs15"
# data_name = "email-enron"
data_name = "ndc-substances"
for i in range(4):
    property_ = properties[i]
    print(property_)
    #ax[i].loglog()
    ax[i].set(title = f"{property_}")
    ax[i].set(xlabel = "Timestep")
    
    count = 0
    
    for jj, ss in enumerate(suffix):
    
        with open('fig_backup/res_atu_{}{}.pkl'.format(data_name, ss), 'rb') as fp:
            pars = pickle.load(fp)

        ll = process_intersect(pars["INTERSECTION"])
        mean_inter = [expectation(x) for x in ll]

        if i in range(3):
            ax[i].plot(timeline, pars[idx[i]][0:100], label = labb[jj], color = inferno[jj])
        elif i == 3:
            ax[i].plot(timeline, mean_inter[0:100], label = labb[jj], color = inferno[jj])
            
    

            
            
fig.suptitle(data_name) 
ax[0].legend() #bbox_to_anchor=(-0.1, 0.7)
#ax[0].set(ylabel = "Edge intersection rate (normalized)")
plt.tight_layout()
plt.savefig("figures/backup_{}_newPA_atu.pdf".format(data_name),bbox_inches='tight')
