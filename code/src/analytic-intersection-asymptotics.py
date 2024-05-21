import numpy as np 
from itertools import product 
from scipy.special import binom


class MatrixConstructor: 
    
    def __init__(self, k_max, pars):
        
        self.k_max = k_max
        self.pars = pars
        
        self.a_array()
        self.b_array()
        
    def a_array(self):
        
        self.A = np.ones((self.k_max, 
                           self.k_max, 
                           self.k_max, 
                           self.k_max, 
                           self.k_max))
        
    def b_array(self): 
        """
        array of coefficients for first term in the linear map
        
        probably wrong in various ways which need to be investigated further and tested
        """
        
        self.B = np.zeros((self.k_max, self.k_max, self.k_max))
        
        eta = self.pars["eta"]
        BETA = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        I = np.arange(self.k_max)[:, None, None]
        J = np.arange(self.k_max)[None, :, None]
        K = np.arange(self.k_max)[None, None, :]
        
        M = binom(J-1, K-1)*eta**(K-1)*(1-eta)**(J-K) 
        M[np.isnan(M)] = 0
        
        for i, j, k in product(range(self.k_max), range(self.k_max), range(self.k_max)):
            
            self.B[i,j,k] = M[:,j,k] * sum(
                BETA[ell]*GAMMA[i-k-ell]
                for ell in range(1, i - k)
            )
            
        
    def linear_map(self, T): 
        self.k_max = T.shape[0]
    # form a tensor for us to populate
        S = np.zeros_like(T)
        
        for i, j, k in product(range(self.k_max), range(self.k_max), range(self.k_max)):
            
            # Case 1: k = 0
            if k == 0: 
                S[i,j,0] = 1/2*sum(
                                  T[ell,j,0]*self.A[i,0,ell,j,0] + 
                                  T[j,ell,0]*self.A[i,0,j,ell,0]
                                  for ell in range(self.k_max)
                                )
            
            # Case 2: k >= 1
            else: 
                S[i,j,k] = 1/2*sum(
                                T[ell,j,k]*self.A[i,k,ell,j,k] + 
                                T[j,ell,k]*self.A[i,k,j,ell,k]
                                for ell, h in product(range(self.k_max), range(self.k_max)) if h >= k
                                )
                
                S[i,j,k] += self.B[i, j, k]*(sum(
                    T[ell,j,k] + T[j,ell,k] for ell in range(self.k_max)
                ))
                    
        return S
    
    def matrix_of_linear_map(self):
        
        I = np.eye(self.k_max**3)
        M = np.zeros((self.k_max**3, self.k_max**3))
        
        for i in range(self.k_max**3):
            
            u = I[:, i]
            T = vec_to_tensor(u, self.k_max)
            S = self.linear_map(T)
            v = tensor_to_vec(S, self.k_max)
            M[:, i] = v

        return M
    
def vec_to_tensor(u, k_max): 
    return u.reshape((k_max, k_max, k_max))
    
def tensor_to_vec(S, k_max): 
    return S.flatten()


