# +
import json
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgi
from decimal import Decimal
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



# +
## code to print the link prediction results for the new model
# For the Randomized Draws, i.e. not temporailiry dependent assumption. 

# print("dataset","&",  "Type", "&", "Num of Nodes","&", "Num of Edges" , "&", "AUC", "&",  "REC", "&", "F1",
#       "&", "AUC (temporal)", "&",  "REC (temporal)", "&", "F1 (temporal)", r"\\" )

print("dataset","&",  "Type", "&", "Num of Nodes","&", "Num of Edges" , "&", "AUC", "&",  "REC",
      "&", "AUC (temporal)", "&",  "REC (temporal)", r"\\" )

incomplete = []

incomplete_random = []

for xx, data_name in enumerate(realworld_Hgraphs):

    H_type = data_type_dict[data_name]
    
    AUC = []
    REC = []
    F1_ = []

    AUC_temporal = []
    REC_temporal = []
    F1__temporal = []

    for jj in range(10):
        try:
            with open('lp_100k_random/{}_CS_{}.pkl'.format(data_name, jj), 'rb') as f:
                d = pickle.load(f)
            #print(d)
            AUC.append(d["AUC"])
            REC.append(d["recall"])
            F1_.append(d["f1"])
        
        except:
            if data_name != "dawn" and data_name != "tags-stack-overflow":
                #print(data_name)
                incomplete_random.append((data_name, jj))
            
    for jj in range(1):
        try:
            with open('lp_100k_temporal/{}_CS_{}.pkl'.format(data_name, jj), 'rb') as f:
                d = pickle.load(f)
            #print(d)
            AUC_temporal.append(d["AUC"])
            REC_temporal.append(d["recall"])
            #F1__temporal.append(d["f1"])

        except:
            incomplete.append((data_name, jj))
    


    if data_name in temporal:
        
        
        print( "\\data{", data_name, "}", "&", H_type, "&", num_nodes[xx], "&", num_edges[xx], "&",  
              format(np.mean(AUC),'.3f'),"&",  format(np.mean(REC),'.3f'), 
              "&", 
              format(np.mean(AUC_temporal),'.3f'),"&",  format(np.mean(REC_temporal),'.3f'),  r"\\")
    
    
    else:
    
        print("\\data{", data_name, "}", "&", H_type, "&", num_nodes[xx], "&", num_edges[xx], "&",  
              round(np.mean(AUC),3),"&",  round(np.mean(REC),3), "&",
              "NA","&",  "NA", r"\\")



