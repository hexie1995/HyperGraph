# %load_ext autoreload
# %autoreload 2
import importlib 
m = importlib.import_module(".model", "src")
em = importlib.import_module(".em", "src")
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

""
real_datasets = ["coauth-mag-geology", "coauth-mag-history", "congress-bills", "contact-high-school", 
                 "contact-primary-school", "email-enron", "email-eu", "hospital-lyon",
                 "ndc-substances", "tags-ask-ubuntu", "tags-math-sx"]
                 
#"diseasome","disgenenet",
real_datasets = [  "tags-stack-exchange"]


""
# Test Link Prediction on Real-world Network

def get_lp_result(timesteps, data_name):

    H0 = xgi.load_xgi_data(data_name)
    H0 = m.GrowingHypergraph(H0)
    longest = max(H0.edges.size.aslist())

    H1 = xgi.Hypergraph()
    for i in range(timesteps):
        H1.add_edge(H0.edges.members(i))

    # Fit the model to get the esitmated parameters for email enron graph
    model1 = em.EM(eslg, nhl, nnl) 
    H1 = m.GrowingHypergraph(H1)
    model1.fit(H1)

    eta1 = model1.pars["eta"]
    beta1 = model1.pars["beta"]
    gamma1 = model1.pars["gamma"]
    model1.pars

    with open('results/res_{}.json'.format(data_name), 'w') as fp:
        json.dump(model1.pars, fp)

    # now we build a model using the guaranteed version of the sampler 
    # because that's how we built the original hypergraph
    pos_edges = []
    neg_edges = []

    for i in range(timesteps, timesteps*2):    
        pos_edges.append(list(H0.edges.members(i)))
        rand = random.randint(2, longest)
        neg_edges.append(random.sample(range(1, len(H1.nodes)), rand))

    to_pred = [list(x) for x in pos_edges] + [list(x) for x in neg_edges]
    true_label = [1]*timesteps + [0]*timesteps
    PRED = model1.predict_v1(to_pred)

    #return PRED, PRED1, PRED2, true_label
    plot_AUC(PRED, true_label, data_name)
    plot_PR(PRED, true_label, data_name)


""
for data in real_datasets:
    get_lp_result(1000, data)

""
import numpy as np

X = [[1,2,3,4],[5,6,7,8], [9,0,1,2]]
X = np.array(X)
print(X)
X[-2:, :2]
