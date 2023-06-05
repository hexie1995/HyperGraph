import xgi
import numpy as np
import copy
import math
import scipy.special as ss
from collections import Counter

# +
class GrowingHypergraph():
    
    def __init__(self, H = None, track_branching = True, store_graphs = True):
        
        """
        Unclear what we want track_branching for, but it might be interesting for things like community detection or visualization. 
        """
        if H:
            H = xgi.classes.function.convert_labels_to_integers(H)
            self.H = H
        else:
            self.H = xgi.Hypergraph()
        
        if track_branching: 
            self.track_branching = True
            self.branches = dict()
            
        if store_graphs:
            self.e_vecs = []
            self.H_vecs = []
            
    
        self.nodes = self.H.nodes    
        self.num_nodes = self.H.num_nodes
        self.num_edges = self.H.num_edges
        self.edges = self.H.edges
        self._hypergraph = self.H._hypergraph 
        
        
        
    def add_edge(self, e, log_branch = True, store_graphs = True):
        if store_graphs:
            H_temp = copy.deepcopy(self.H)
            self.H_vecs.append(H_temp)
        self.H.add_edge(e)
        if log_branch: 
            self.branches.update({self.H.num_edges -1: -1}) 
        if store_graphs:
            self.e_vecs.append(e)
            
    
    def sample_edge(self, eta, gamma, beta, force_one_node = False):
        
        e_ = set()
        
        while len(e_) <= 0:
            # first, sample existing edge
            ix = np.random.randint(self.H.num_edges)
            
            # if we are tracking the branching process
            # log that edge about to be formed was formed from edge ix. 
            if self.track_branching:
                self.branches.update({self.H.num_edges + 1 : ix})
            
            e = self.H.edges.members(ix)
            
            if force_one_node: 
                i = np.random.choice(list(e))
                e_.add(i)
            
            for i in e:
                if np.random.rand() < eta:
                    e_.add(i)
            
            # then, add unrelated nodes from graph
            num_left   = self.H.num_nodes - len(e)
            num_to_add = np.random.binomial(num_left, gamma/self.H.num_nodes)
            
            num_added = 0
            
            while num_added < num_to_add: 
                candidate = np.random.randint(0, self.H.num_nodes)
                # candidate = np.random.choice(self.H.nodes, 1)[0]
                if candidate not in e:
                    e_.add(candidate)
                    num_added += 1
            
            # then, add completely novel nodes
            num_to_add = np.random.poisson(beta)
            for i in range(self.H.num_nodes, self.H.num_nodes + num_to_add):
                e_.add(i)
        
        self.add_edge(e_, log_branch = False)
        
        
    def sample_edge_v1(self, eta, gamma, beta, force_one_node = False):
        
        e_ = set()
        
        while len(e_) <= 0:
            # first, sample existing edge
            ix = np.random.randint(self.H.num_edges)
            e = self.H.edges.members(ix)
            
            if force_one_node: 
                i = np.random.choice(list(e))
                e_.add(i)
            
            for i in e:
                if np.random.rand() < eta :
                    e_.add(i)
            
            # then, add unrelated nodes from graph
            node_left   =  self.H.nodes-e

            for i in node_left:
                if np.random.rand() < gamma :
                    e_.add(i)
            
            
            # then, add completely novel nodes
            num_to_add = np.random.poisson(beta)
            for i in range(self.H.num_nodes, self.H.num_nodes + num_to_add):
                e_.add(i)
        
        self.add_edge(e_)
        return e_
    
    def edge_predecessor(self, edge_ix):
        """
        return the index of the edge used to sample the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        else:
            return self.branches[edge_ix]
    
    def edge_ancestor(self, edge_ix):
        """
        return the root of the branching tree containing the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        
        ix = edge_ix
        while self.branches[ix] != -1:
            ix = self.branches[ix]
        return ix
          
    def edge_lineage(self, edge_ix):
        """
        sequence of edges from the supplied edge index to the root of the branching tree containing the supplied edge index
        """
        if not self.track_branching: 
            raise(ValueError, "This model was not initialized to track the edge branching process.")
        
        ix = edge_ix
        lineage = [ix]
        
        while self.branches[ix] != -1:
            ix = self.branches[ix]
            lineage.append(ix)
        
        return lineage 
        
    def degree_sequence(self):
        return self.H.nodes.degree.asnumpy()
    
    def edge_size_sequence(self):
        return self.H.edges.size.asnumpy()
    
    def rw_laplacian(self, **kwargs):
        A = xgi.adjacency_matrix(self.H, **kwargs)
        D = np.diag(A.sum(axis = 1))
        D_inv = np.linalg.inv(D)
        L = D_inv @ (D - A)
        return L
    
    def intersection_profile(self, num_samples = int(1e6)):
        
        edge_sizes = [len(self.H.edges.members(e)) for e in self.H.edges]
        # min_edge_size = min(edge_sizes)
        max_edge_size = max(edge_sizes)
        
        C = np.zeros((max_edge_size+1, max_edge_size+1, max_edge_size+1))
        
        for _ in range(num_samples):
            i, j = 0, 0
            
            while i == j: 
                i = np.random.randint(self.H.num_edges)
                j = np.random.randint(self.H.num_edges)
            
            e = self.H.edges.members(i)
            f = self.H.edges.members(j)
            
            k = len(e.intersection(f))
            
            C[len(e), len(f), k] += 1
            C[len(f), len(e), k] += 1
            
        return C / (2*num_samples)
    
    
    def EM_update_vec(self, eta, gamma, beta, max_iter, H_vec = None, e_vec = None, use_self = True):


        # note that e_vec is now a vector with the sequence of added edges over time
        # here H and max_iter stays the same as before, with H being the initialized hypergraph, and max_iter the iteration of update
        # the averaged output of the first iteration will be the guess of the next iteration, so on and so forth.
        # eta, gamma, beta are all vectorized.

        
        if use_self:
            H_vec = self.H_vecs
            e_vec = self.e_vecs
            


        total_edge = H_vec[-1].num_edges

        heta, hgamma, hbeta = eta, gamma, beta

        #heta_vec = np.full((total_num_new, total_edge), np.mean(heta))
        #hgamma_vec = np.full((total_num_new, total_edge), np.mean(hgamma))
        #hbeta_vec = np.full((total_num_new, total_edge), np.mean(hbeta))

        #print(heta_vec.shape)

        hprev = -np.inf

        #intialize error term
        true_error = 1

        KVEC = []
        KPVEC = []
        NPVEC = []
        NMUVEC = []
        BVEC = []
        NUM_EDGES = []

        for H,e in zip(H_vec, e_vec):

            #facts about the H graph
            num_of_edges = H.num_edges
            N = H.num_nodes

            #facts about the added edge
            k_vec = []
            kp_vec = []
            Np_vec = []
            Nmu_vec = []
            b_vec = []

            #calculate necessary vectors for EM
            for i in range(num_of_edges):
                e_i = H.edges.members(i)
                N_mu = len(e_i)
                k = N_mu - len(e.intersection(e_i))
                Np  = N - N_mu
                k_prime = len(e.intersection(H.nodes-e_i))
                new_added = len(e.difference(H.nodes))


                k_vec.append(k)
                kp_vec.append(k_prime)
                Np_vec.append(Np)
                Nmu_vec.append(N_mu)
                b_vec.append(new_added)


            if num_of_edges<total_edge:
                for _ in range(total_edge-num_of_edges):
                    k_vec.append(0)
                    kp_vec.append(0)
                    Np_vec.append(0)
                    Nmu_vec.append(0)
                    b_vec.append(new_added)
            # these are all numpy array matrix, list of lists, row = how many edges added, column = how many edges are in each time

            KVEC.append(k_vec)
            KPVEC.append(kp_vec)
            NPVEC.append(Np_vec)
            NMUVEC.append(Nmu_vec)
            BVEC.append(b_vec)

            # this is just a list, their length = how many edges added
            NUM_EDGES.append(num_of_edges)

        #print(KVEC)

        KVEC = np.array(KVEC)
        KPVEC = np.array(KPVEC)
        NPVEC = np.array(NPVEC)
        NMUVEC = np.array(NMUVEC)
        BVEC = np.array(BVEC)
        NUM_EDGES = np.array(NUM_EDGES)

        #print(BVEC)

        ######## Enter the vectorized iteration ########

        correct = []


        count = 0
        while count<max_iter:


            PZXVEC, total_vec, bscore_vec = self.E_step_vec(heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES)
            cor= self.Test_Correct_vec(NUM_EDGES, total_vec, bscore_vec) 
            heta, hgamma, hbeta = self.M_step_vec(PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES) 

            count = count+1
            correct.append(cor)


            #print("HETA", heta_vec)
            #print(hgamma_vec)
            #print(hbeta_vec)


        #return heta, hgamma, hbeta
        return heta, hgamma, hbeta, correct

    def E_step_vec(self, heta, hgamma, hbeta, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):

        s1 = np.power((1-heta), KVEC)
        s2 = np.power(heta, (NMUVEC- KVEC))
        s3 = np.power((1-hgamma), (NPVEC - KPVEC))
        s4 = np.power(hgamma, KPVEC)


        #print(s1)

        # this step turn hbeta into a matrix that has the same dimension as the others 
        # however this matrix should be the same on each row 
        bscore_vec = (np.power(hbeta, BVEC)*np.exp(hbeta))/(ss.factorial(BVEC))

        #print(bscore_vec)

        # this step will return a matrxi num_of_added_edge(i.e time steps T) x num_of_edges at the last time step (i.e. H_n.num_edges) 
        # and the upper left corner of this matrix will be all 0 because those are just the missing information from each time slot
        # and that will be okay because the corresponding k is also 0 there
        # dimension of pzx and KVEC shoud be exactly the same.

        pzx = s1*s2*s3*s4
        #print(pzx)

        pzx_vec= []        
        total_vec=[]


        # this is a necessary step to calculate for each ROW of pzx the summation and the averaged value. 
        for i,ne in enumerate(NUM_EDGES):

            total = sum(pzx[i][0:ne])
            total_vec.append(total)
            pzx_vec.append([x/total for x in pzx[i]])

        PZXVEC = np.array(pzx_vec)

        #print(PZXVEC)
        return PZXVEC, total_vec, bscore_vec

    def M_step_vec(self, PZXVEC, KVEC, KPVEC, NPVEC, NMUVEC, BVEC, NUM_EDGES):

        heta =  1-(sum(np.einsum('ij, ij->i', KVEC, PZXVEC))/sum(np.einsum('ij, ij->i',NMUVEC, PZXVEC)))
        hgamma = sum(np.einsum('ij, ij->i',KPVEC, PZXVEC))/sum(np.einsum('ij, ij->i', NPVEC, PZXVEC))

        # we only consider poisson distribution for now, the optimal beta is equvilant to lambda in poisson.
        hbeta = sum(np.einsum('ij, ij->i',BVEC, PZXVEC))/np.sum(PZXVEC)
        #print(hbeta)

        return heta, hgamma, hbeta


    def Test_Correct_vec(self, NUM_EDGES, total_vec, bscore_vec):

        curr_vec = np.sum(np.log(total_vec) + np.log(bscore_vec[:,0]) -np.log(NUM_EDGES))

        return curr_vec

    

    def Test_error_random_graph(self, parameters):
        
        # parameter consists of 6 different values, true eta and initiliazed eta and etc. 
        # return a list of marginal log likelihood and thus will see how fast it converges
        
        
        H0 = xgi.random_hypergraph(50, [0.1, 0.01])
        eta, gamma, beta, ieta,igamma,ibeta = parameters

        eta_temp,gamma_temp,beta_temp = ieta,igamma,ibeta

        eta_vec= []
        gamma_vec=[]
        beta_vec=[]
        cor_vec = []

        H_temp = H0

        e_vec = []
        H_vec = []
        # test a hypergraph that 100 edges has been added since the initial state
        # each iteration of the heta and hbeta are averaged out over 100 runs of the EM itself


        for _ in range(100):
            H_vec.append(H_temp)
            H1 = GrowingHypergraph(H_temp)
            e_ = H1.sample_edge_v1(eta, gamma, beta, force_one_node = False)        
            H_temp = H1.H
            e_vec.append(e_)


        eta_vec,gamma_vec,beta_vec,cor_vec = self.EM_update_vec(eta_temp,gamma_temp, beta_temp,100, H_vec, e_vec, use_self = False)
  


        with open("error_vecs.txt", "a") as f:
            print(np.mean(eta_vec)-eta ,np.mean(gamma_vec)-gamma,np.mean(beta_vec)-beta, file=f)

        #np.save("results/" + "ETA_" + str(parameters) + ".npy", eta_vec)      
        #np.save("results/" + "GAMMA_" + str(parameters) + ".npy", gamma_vec)      
        #np.save("results/" + "BETA_" + str(parameters) + ".npy", beta_vec)      
        np.save("results/" + "COR_" + str(parameters) + ".npy", cor_vec)      

        return cor_vec    

    
    
    
    
    
    
# -


