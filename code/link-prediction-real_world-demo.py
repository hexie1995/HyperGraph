# +
# %load_ext autoreload
# %autoreload 2
import importlib 
m = importlib.import_module(".model", "src")
em = importlib.import_module(".em", "src")
sem = importlib.import_module(".sem", "src") 

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import xgi
import copy
import random
import statistics
import matplotlib.pyplot as plt
import scipy.special as ss
import json

# hypergraphs with less than 50k edges are considered "small" and could use exact EM directly

small_realworld_Hgraphs = ["email-enron", "diseasome", "hospital-lyon"]

realworld_Hgraphs = ["coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "email-enron", "email-eu", "hospital-lyon",
                 "ndc-substances", "tags-ask-ubuntu", "tags-math-sx" ,  "diseasome","disgenenet", "tags-stack-exchange"]


# -

# # Plotting Functions for the ROC and PR

def plot_AUC(y_pred, y_true, plot_name):
    
    fpr, tpr, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.title('ROC curve ' + plot_name)
    plt.legend(loc="lower right")
    plt.savefig("figures/ROC_{}.pdf".format(plot_name), bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_PR(y_pred, y_true, plot_name):
    
    plt.clf()
    plt.figure()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, marker='.', label='PR curve')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title('Precision Recall '+plot_name)
    plt.savefig("figures/PR_{}.pdf".format(plot_name), bbox_inches='tight')
    plt.show()
    plt.clf()

# estimator likelihood definitions
# pmfs are used in E-step
# m_step estimators are the parameter estimators for each distribution
# used in m-step, using the matrix chi formed in the E-step

# edge sampling
def edge_sample_pmf(x, t, eta):
    return (eta**x)*((1-eta)**(t-x))

def edge_sample_m_step(x, t, chi):
    C = np.tril(chi,-1)
    return (x*C).sum() / (t*C).sum()

esl = em.EdgeSampleLikelihood(
    pmf = edge_sample_pmf, 
    m_step = edge_sample_m_step
    )

# edge sampling: guaranteed version
def guaranteed_edge_sample_pmf(x, t, eta):
    M = (eta**(x-1))*((1-eta)**(t-x)) # ?
    M[x == 0] = 0
    return M

def guaranteed_edge_sample_m_step(x, t, chi):
    C = np.tril(chi,-1)
    top = ((x-1)*C).sum()
    bottom = ((t-1)*C).sum()
    return top / bottom

eslg = em.EdgeSampleLikelihood(
    pmf = guaranteed_edge_sample_pmf, 
    m_step = guaranteed_edge_sample_m_step
    )

# addition of novel nodes not previously seen in the hypergraph
def novel_nodes_pmf(k, beta):
     return (beta**k)*np.exp(beta)/(ss.factorial(k))

def novel_nodes_m_step(k, chi):
    return k.mean()

nnl = em.NovelNodesLikelihood(pmf = novel_nodes_pmf, m_step = novel_nodes_m_step)

# addition of nodes from rest of hypergraph
def nodes_from_hypergraph_pmf(k, gamma):
    return (gamma**k)*np.exp(-gamma)/(ss.factorial(k))

def nodes_from_hypergraph_m_step(x, chi):
    return (np.tril(chi, -1)*x).sum(axis = 1).mean()

nhl = em.NodesFromHypergraphLikelihood(pmf = nodes_from_hypergraph_pmf, m_step = nodes_from_hypergraph_m_step)


#
# # Test Link Prediction on Real-world Network

# ## This function sample negative edges according to edge size distribution and degree distribution

def generate_negative_samples(H, num):
    
    
    # 1. sample the edge size unfiorm randomly from the entire set of edges
    # 2. sample nodes based on the degree sequence of the nodes 
    
    neg_edges = []
    
    for i in range(num):
        
        edge_size = random.choice(H.edges.size.asnumpy()) 
        deg_seq = list(H.nodes.degree.asnumpy())
        normalized_deg_seq =  [float(x)/sum(deg_seq) for x in deg_seq]
        neg_edges.append(np.random.choice(list(H.nodes), edge_size, p=normalized_deg_seq))
    
    return neg_edges
    

# ## This function generate the base graph H1 (first 80% of the Hypergraph timestamp) and save the rest 20% of hyper  edges as the test postive set, + the negative edges sampled earlier and produce a label for everyone

def generate_true_false_edges(data_name):

    # 1. There are multiple ways to to do this, but right now, it is assumed that each edge's probability are calculated independently
    # 2. If a dataset has more than 50k edges, we skip to SEM
    # 3. The number of positive edges sampled from each network is settled to be 1000 for now.
    
    
    
    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())

    
    total_timesteps = H0.num_edges
    train_timesteps = round(H0.num_edges*0.8)
    test_timesteps = total_timesteps - train_timesteps
    
    
    H1 = xgi.Hypergraph()
    for i in range(train_timesteps):
        H1.add_edge(H0.edges.members(i))

    H1 = m.GrowingHypergraph(H1)
    
    # now we build a model using the guaranteed version of the sampler 
    # because that's how we built the original hypergraph
    neg_edges = generate_negative_samples(H1.H, test_timesteps)
    pos_edges = []
    
    for i in range(train_timesteps, total_timesteps):    
        pos_edges.append(list(H0.edges.members(i)))
        
    to_pred = [list(x) for x in pos_edges] + [list(x) for x in neg_edges]
    true_label = [1]*test_timesteps + [0]*test_timesteps
    
    return H0, H1, to_pred, true_label


# ## This function uses the previous information and generate the exact link prediction with exact EM

def exactEM_link_prediction_plot(data_name, H1, edges_pred, labels):
    
   # this function only works for small hypergraphs

    model1 = em.EM(eslg, nhl, nnl) 
    model1.fit(H1)

    #eta1 = model1.pars["eta"]
    #beta1 = model1.pars["beta"]
    #gamma1 = model1.pars["gamma"]
    #model1.pars

    with open('results/res_{}.json'.format(data_name + "_exact"), 'w') as fp:
        json.dump(model1.pars, fp)


    PRED = model1.predict_v1(edges_pred)

    #return PRED, PRED1, PRED2, true_label
    plot_AUC(PRED, labels, data_name + "_exact")
    plot_PR(PRED, labels, data_name + "_exact")


# ## This function uses SEM/SEM_VR to predict links

def SEMvr_link_prediction_plot(data_name, steps, H0, H1, edges_pred, labels):
    
   # this function only works for small hypergraphs

    EM = sem.SEM(H1, pars = {"eta"   : 0.3, 
                            "beta"  : 0.4, 
                            "gamma" : 0.02})

    # change this according to your wish
    steps = 5000

    for i in range(steps):

        if i % 250 == 0:
            print("--------")
            print(f"Completed step {i}")
            print(EM.pars)
        
        # TODO: change rho to be a decaying learning rate/finetune it.
        EM.SEM_step(0.002)
    
    
    with open('results/res_{}.json'.format(data_name + "_SEM"), 'w') as fp:
        json.dump(EM.pars, fp)

    PRED = []
    for edge in edges_pred:
        PRED.append(EM.predict(H0, edge, EM.pars))
    
    np.array
    
    
    #return PRED, PRED1, PRED2, true_label
    plot_AUC(PRED, labels, data_name + "_SEM")
    plot_PR(PRED, labels, data_name + "_SEM")

# ## The following is a demo for the dataset "email-enron"

H0, H1, edges_pred, labels = generate_true_false_edges("email-enron")

exactEM_link_prediction_plot("email-enron", H1, edges_pred, labels)

SEMvr_link_prediction_plot("email-enron", 5000 , H0, H1, edges_pred, labels)
