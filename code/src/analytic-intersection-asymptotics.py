import numpy as np 
from itertools import product 
from scipy.special import binom

class MatrixConstructor: 
    
    def __init__(self, k_max, pars):
        
        self.k_max = k_max
        self.pars = pars
        
        self.b_array()
        # self.omega_array()
        # self.a_array()
        self.a_and_omega_arrays()
    
    def a_and_omega_arrays(self):
    
        k_max = self.k_max
        eta   = self.pars["eta"]
        BETA  = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        mu_BETA = expectation(BETA)
        
        # initialize all the array indices
        I = np.arange(k_max)[:,None,None,None,None,None,None]
        K = np.arange(k_max)[None,:,None,None,None,None,None]
        L = np.arange(k_max)[None,None,:,None,None,None,None]
        J = np.arange(k_max)[None,None,None,:,None,None,None]
        H = np.arange(k_max)[None,None,None,None,:,None,None]
        S = np.arange(k_max)[None,None,None,None,None,:,None]
        X = np.arange(k_max)[None,None,None,None,None,None,:]
        
        # used in both arrays
        
        # note: although indexed sljh, it is correct for this not to depend on h
        
        t1_sljh = binomial(successes = S-1, trials = L-1, prob = eta)
        t1_sljh[np.isnan(t1_sljh)] = 0
        t1_sljh = np.tile(t1_sljh, (k_max, k_max, 1, k_max, k_max, 1, k_max)) 


        # used in both arrays
        gamma_x = GAMMA[X]
        gamma_x = np.tile(gamma_x, (k_max, k_max, k_max, k_max, k_max, k_max, 1))
        
        # used in both arrays
        ix = I - S - X
        mask = (ix < 0).copy()
        ix[mask] = 0   # required for indexing into BETA without error
                       # need to later go and zero out the results
        t3_isx = BETA[ix]
        t3_isx[mask] = 0
        
        # now we are ready to tile
        t3_isx = np.tile(t3_isx, (1, k_max, k_max, k_max, k_max, 1, 1))
        
        # used only in A
        # hypergeometric(k_success = K, k = Y, n_success = H, n = L)
        #  note: need to zero out the case that K > J
        w1_kslh = hypergeometric(k_success = K, k = S, n_success = H, n = L)
        w1_kslh[np.isnan(w1_kslh)] = 0
        w1_kslh = np.tile(w1_kslh, (k_max, 1, 1, k_max, 1, 1, k_max)) 
        mask = np.tile(K > J, (k_max, 1, k_max, 1,  k_max, k_max, k_max))
        w1_kslh[mask] = 0
        mask = np.tile(H > J, (k_max, k_max, k_max, 1,  1, k_max, k_max))
        w1_kslh[mask] = 0
        
        # used only in OMEGA
        w2_ksxljh = hypergeometric(k_success = K-1, k = S, n_success = H, n = L)*X*(J - K + 1)*mu_BETA
        w2_ksxljh[np.isnan(w2_ksxljh)] = 0
        w2_ksxljh = np.tile(w2_ksxljh, (k_max, 1, 1, 1, 1, 1, 1))
        mask = np.tile(K > J + 1, (k_max, 1, k_max, 1, k_max, k_max, k_max))
        w2_ksxljh[mask] = 0
        
        # store arrays as instance variables
        
        # labeled phi in the writeup
        self.A     = (t3_isx*gamma_x*w1_kslh*t1_sljh).sum(axis = (5, 6))
                
        # labeled psi in the writeup
        self.OMEGA = (t3_isx*gamma_x*w2_ksxljh*t1_sljh).sum(axis = (5, 6))
       
    def b_array(self): 
        """
        array of coefficients for first term in the linear map
        """
        
        k_max = self.k_max
        
        self.B = np.zeros((k_max, k_max, k_max))
        
        eta   = self.pars["eta"]
        BETA  = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        K = np.arange(k_max)[:, None]
        J = np.arange(k_max)[None, :]
        
        # first term: probability of intersection of size k
        # THIS would be a good place to check for issues
        M = binomial(successes = K-1, trials = J-1, prob = eta)
        mask = (K > J) | (J == 0) 
        M[mask] = 0
        # self.M = M
        
        # second term: probability of edge of size i given intersection of size k
        
        # any way we introduce some double counting here or below?
        N = np.zeros((k_max, k_max))
        for i, k in product(range(1, k_max), range(1, k_max)):
            N[i, k] = sum(
                BETA[x]*GAMMA[i-k-x]
                for x in range(0, i - k + 1)
            )
        
        for i, k, j in product(range(k_max), range(k_max), range(k_max)):
            self.B[i,k,j] = M[k,j] * N[i,k]
        
        
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
        
    def linear_map(self, T): 
        
        S = np.zeros_like(T)
        
        # case 1: k = 0
        # this CREATES k = 0 cases in the output tensor S
        # it does so by *also* using h = 0 cases in the input tensor T
        # numerically, this term can play quite a dominant role in the 
        # spectral structure
        # this term seems reliable, in the sense that in combination with the third term it captures the density of very large intersections nearly exactly
        # NOTE: adding a factor of 1 (instead of 1/2) here does change the eigenvalue of the matrix
        
        # eq. 81 in current draft
        
        # empirically   correct coef: 1/2
        # theoretically correct coef: 1/2
        to_add = 1/2*(
            np.einsum("lj,ilj -> ij", T[1:,1:,0], self.A[1:,0,1:,1:,0]) + 
            np.einsum("jl,ijl -> ij", T[1:,1:,0], self.A[1:,0,1:,1:,0])
        )
        S[1:,1:,0] = to_add
        
        # case 2: k >= 1
        ## term 1
        # this CREATES intersections FROM other intersections, not from h = 0 cases. 
        # "intersection decay"
        
        # Should have a coefficient of 1 in front of it using the arguments in the current draft...? So now we need to justify the presence of the 1/2 here. 
        
        # empirically   correct coef: 1/2? 1?
        # theoretically correct coef: 1
        to_add = 1*(
            np.einsum("ljh,ikljh -> ijk", T[1:,1:,:], self.A[1:,1:,1:,1:,:]) + 
            np.einsum("jlh,ikjlh -> ijk", T[1:,1:,:], self.A[1:,1:,1:,1:,:])
        )
        S[1:,1:,1:] += to_add
        
        ## term 2
        # this USES  h = 0 cases in the input tensor T
        # this term seems reliable, in the sense that it captures the 
        # density of very large intersections nearly exactly. 
        # this is the term for which the factor of 1/2 is in question
        # both with and without this factor, we replicate the distribution 
        # of edge sizes very precisely.
        # with the factor of 1/2, we are off by roughly a factor of 2 in the tail of the distribution.
        
        # eq. 79/80, first term (with b_{ik|j})
        
        # empirically   correct coef: 1
        # theoretically correct coef: 1
        to_add = 1*(
            np.einsum("lj,ikj -> ijk", T[1:, 1:, 0], self.B[1:,1:,1:]) +
            np.einsum("jl,ikj -> ijk", T[1:, 1:, 0], self.B[1:,1:,1:])
            )
        
        S[1:,1:,1:] += to_add
        
        ## term 3 
        # this USES h = 0 cases in the input tensor in order to create intersections of size k = 1 through the extant node addition mechanism. 
        
        # eq. 79, 2nd term in current draft
        
        # empirically   correct coef: 1/2
        # theoretically correct coef: 1
        
        # OMEGA is indexed ik|ljh
        to_add = 1*(
            np.einsum("lj,ilj -> ij", T[1:, 1:, 0], self.OMEGA[1:,1,1:,1:,0]) + 
            np.einsum("jl,ijl -> ij", T[1:, 1:, 0], self.OMEGA[1:,1,1:,1:,0]) 
        )
        
        S[1:,1:,1] += to_add

        return S
    
    def matrix_of_linear_map(self):
        
        k_max = self.k_max
        
        I = np.eye(k_max**3)
        M = np.zeros((k_max**3, k_max**3))
        
        for i in range(k_max**3):
            
            u = I[:, i]
            T = vec_to_tensor(u, k_max)
            S = self.linear_map(T)
            v = tensor_to_vec(S, k_max)
            
            M[:, i] = v

        return M
    
    def linear_map_on_zeros(self, T):
        """
        just for testing
        purpose is to test the entries of the matrix self.A corresponding to zero intersections h and k
        I think that the linear map defined by this segment of the matrix should be stochastic. 
        """
         
        return 1/2*(
            np.einsum("lj,ilj -> ij", T[:, :], self.A[:, 0, :, :, 0]) + 
            np.einsum("jl,ijl -> ij", T[:, :], self.A[:, 0, :, :, 0])
        )
        
    def matrix_of_map_on_zeros(self):
        
        k_max = self.k_max
        
        I = np.eye(k_max**2)
        M = np.zeros((k_max**2, k_max**2))
        
        for i in range(k_max**2):
            
            u = I[:, i]
            T = u.reshape(k_max, k_max)
            S = self.linear_map_on_zeros(T)
            v = S.reshape(k_max**2)
            M[:, i] = v

        return M
    
def vec_to_tensor(u, k_max): 
    return u.reshape((k_max, k_max, k_max))
    
def tensor_to_vec(S, k_max): 
    return S.flatten()

def expectation(x): 
    return np.sum(x*np.arange(len(x)))

def hypergeometric(k_success, k, n_success, n): 
    """
    hypergeometric distribution
    """
    return binom(n_success, k_success)*binom(n - n_success, k - k_success)/binom(n, k)
    
def binomial(successes, trials, prob):
    return binom(trials, successes)*prob**(successes)*(1 - prob)**(trials - successes)    