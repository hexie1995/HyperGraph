import json
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgi
# disgenenet, always encountered devision by 0, something finiky here

plt.style.use('seaborn-v0_8-whitegrid')

def expectation(x):
    
    to_sum = [(i)*j for i,j in enumerate(x)]
      
    return sum(to_sum)

realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "dawn", "disgenenet",
                     "diseasome", "kaggle-whats-cooking", "ndc-classes", "ndc-substances",
                     "tags-ask-ubuntu", "tags-math-sx" , "tags-stack-overflow", "threads-ask-ubuntu", 
                     "threads-math-sx", "threads-stack-overflow",
                     "congress-bills", "contact-high-school", "contact-primary-school",
                     "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", 
                     "invs13", "invs15",  "malawi-village", "science-gallery", "sfhh-conference"]



num_nodes = ["1,930,378", "1,261,129", "1,034,876", "2,558", "12,368", "516", "6,714", "1,161", "5,556",
            "3,029", "1,629", "49,998", "125,602", "176,445", "2,675,969",
            "1,718", "327", "242", "148", "1,005", "75", "113", "92", "232", "86", "10,972", "403"]

num_edges = ["3,700,681", "1,590,335", "1,812,511", "2,272,433", "2,261", "903", "39,774", "49,726", "112,405",
            "271,233", "822,059", "14,458,875", "192,947", "719,792", "11,305,356",
            "282,049", "172,035", "106,879", "10,885", "235,263", "27,834", "19,036",
            "9,644", "73,822", "99,942", "338,765", "54,305"]

#"dawn"
# "tags-stack-overflow"




data_type_dict    = {"coauth-dblp": "co-author", 
                     "coauth-mag-geology": "co-author",
                     "coauth-mag-history": "co-author", 
                     "congress-bills": "social", 
                     "contact-high-school": "social",
                     "contact-primary-school": "social",
                     "dawn": "biological",
                     "diseasome": "biological", 
                     "disgenenet": "biological",
                     "email-enron": "social", 
                     "email-eu": "social",
                     "hospital-lyon": "social",
                     "hypertext-conference": "social",
                     "invs13": "social", 
                     "invs15": "social", 
                     "kaggle-whats-cooking": "biological", 
                     "malawi-village": "social",
                     "ndc-classes": "biological",
                     "ndc-substances": "biological", 
                     "science-gallery": "social", 
                     "sfhh-conference": "social",
                     "tags-ask-ubuntu": "webpage", 
                     "tags-math-sx": "webpage",
                     "tags-stack-overflow": "webpage",
                     "threads-ask-ubuntu": "webpage",
                     "threads-math-sx": "webpage", 
                     "threads-stack-overflow": "webpage"}

nontemporal = ["disgenenet", "hypertext-conference", "invs13", "invs15", 
               "kaggle-whats-cooking", "malawi-village", "science-gallery", "sfhh-conference"]
temporal = list(set(realworld_Hgraphs)-set(nontemporal))

mymarkers = ["o", "v","^","<",">","1","2","3","4","s","p","P","*","h","H","+","x","X","D","d"]


data_type_color = {"co-author": "sandybrown", 
                   "social": "cornflowerblue",
                  "biological": "palevioletred",
                  "webpage": "olivedrab"}

data_marker_dict    = {"coauth-dblp": "o", 
                     "coauth-mag-geology": "v",
                     "coauth-mag-history": "^", 
                     "congress-bills": "o", 
                     "contact-high-school": "v",
                     "contact-primary-school": "^",
                     "diseasome": "o", 
                     "dawn": ">",
                     "disgenenet": "1", 
                     "email-enron": "<", 
                     "email-eu": ">",
                     "hospital-lyon": "*",
                     "hypertext-conference": "D",
                     "invs13": "d", 
                     "invs15": "H", 
                     "kaggle-whats-cooking": "v", 
                     "malawi-village": "s",
                     "ndc-classes": "^",
                     "ndc-substances": "<", 
                     "science-gallery": "p", 
                     "sfhh-conference": "P",
                     "tags-ask-ubuntu": "o", 
                     "tags-math-sx": "v",
                     "tags-stack-overflow": "^",
                     "threads-ask-ubuntu": "<",
                     "threads-math-sx": ">", 
                     "threads-stack-overflow": "1"}


# ## code to read in and output CSV files


print("dataset", "&", "eta", "&",  "beta", "&", "gamma", r"\\" )
ETA = []
BETA = []
GAMMA = []

with open('figures/parameters.csv','w', newline='') as file:

    writer = csv.writer(file)
    
    writer.writerow(["dataset", "eta", "beta", "gamma"])
    #for data_name in ["email-enron"]:
    for data_name in realworld_Hgraphs:
        
        try:
            with open('sem_results_new/res_{}.pkl'.format(data_name + "_SEM_new"), "rb") as f:
                d = pickle.load(f)

            print(d["dataset"],"&",  round(np.mean(d["eta"][-50:]),3),"&",  round(expectation(np.mean(d["beta"][-50:], axis=0)), 3), "&", round(expectation(np.mean(d["gamma"][-50:], axis=0)), 3), r"\\")

            writer.writerow([d["dataset"],  round(np.mean(d["eta"][-50:]),3), round(expectation(np.mean(d["beta"][-50:], axis=0)), 3),  round(expectation(np.mean(d["gamma"][-50:], axis=0)), 3)])
            ETA.append(round(np.mean(d["eta"][-50:]),3))
            BETA.append( round(expectation(np.mean(d["beta"][-50:], axis=0)), 3))   
            GAMMA.append(round(expectation(np.mean(d["gamma"][-50:], axis=0)), 3))


        except:
            print("not finished")

ETA = []
BETA = []
GAMMA = []

for data_name in realworld_Hgraphs:


    with open('sem_results_new/res_{}.pkl'.format(data_name + "_SEM_new"), "rb") as f:
        d = pickle.load(f)

    ETA.append(np.mean(d["eta"][-50:]))
    BETA.append(expectation(np.mean(d["beta"][-50:], axis=0)))   
    GAMMA.append(expectation(np.mean(d["gamma"][-50:], axis=0)))



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
#ax.scatter(BETA, ETA)

for i, txt in enumerate(realworld_Hgraphs):
    ax[0].scatter(np.log(BETA[i]), np.log(1-ETA[i]), 
               marker= data_marker_dict[txt], 
               color = data_type_color[data_type_dict[txt]],
               label = txt)
    #print(BETA[i], np.log(BETA[i]))

    
for i, txt in enumerate(realworld_Hgraphs):
    ax[1].scatter(np.log(GAMMA[i]),np.log(1-ETA[i]), 
               marker= data_marker_dict[txt], 
               color = data_type_color[data_type_dict[txt]],
               label = txt)
    #print(GAMMA[i], np.log(GAMMA[i]))

ax[0].set_xlabel(r"log$(\beta)$")
ax[0].set_ylabel(r"log$(1-\eta)$")
ax[1].set_xlabel(r"log$(\mathbb{E}(\gamma))$")
#ax[1].set_ylabel("ETA")
ax[1].legend(ncol = 2,fontsize=10,  bbox_to_anchor=(1,1))

#ax.set_xscale('log')
#ax.set_title("$\beta$ against $\eta$ ratio")
plt.savefig("figures/Fig5-PARAMETERS.pdf",bbox_inches='tight')


