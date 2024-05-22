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
        """
        probably a pretty inefficient implementation
        """
        
        k_max = self.k_max
        eta = self.pars["eta"]
        BETA = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        self.A = 0.00*np.ones((k_max, 
                           k_max, 
                           k_max, 
                           k_max, 
                           k_max))
        
        I = np.arange(k_max)[:, None, None, None, None, None]
        J = np.arange(k_max)[None, :, None, None, None, None]
        K = np.arange(k_max)[None, None, :, None, None, None]
        L = np.arange(k_max)[None, None, None, :, None, None]
        H = np.arange(k_max)[None, None, None, None, :, None]
        Y = np.arange(k_max)[None, None, None, None, None, :]
        
        # first term
        S1_iy = np.zeros((k_max, k_max))
        for i, y in product(range(k_max), range(k_max)):
            S1_iy[i, y] = sum(BETA[x]*GAMMA[i - y - x] for x in range(0, i - y))
        
        # reshape 
        S1_iy = S1_iy[:, None, None, None, None, :]
        S1_iy = np.tile(S1_iy, (1, k_max, k_max, k_max, k_max, 1)) 

        # create mask and reshape
        mask = Y > I 
        mask = np.tile(mask, (1, k_max, k_max, k_max, k_max, 1))
        S1_iy[mask] = 0
        
        print(np.isnan(S1_iy).mean())
        
        # second term (might not need reshaping, but check)
        S2_kyljh = binom(H, K)*binom(L - H, Y - K)/binom(L, Y)
        # S2_kyljh = S2_kyljh[None, None, :, :, :, :]
        S2_kyljh = np.tile(S2_kyljh, (k_max, k_max, 1, 1, 1, 1))
        
        mask = (K > H) | (Y > L) | ( Y - K > L - H) | (H > L) | (L == 0) 
        mask = np.tile(mask, (k_max, k_max, 1, 1, 1, 1))
        # print(mask.shape)
        # print(S2_kyljh.shape)
        # print(mask.shape)
        # need to reshape
        S2_kyljh[mask] = 0
        
        print(np.isnan(S2_kyljh).mean())
        
        # third term
        S3_yljh = Y/L * binom(L-1, Y-1) * eta**(Y-1) * (1-eta)**(L-Y)
        S3_yljh = np.tile(S3_yljh, (k_max, k_max, k_max, 1, k_max, 1))
        
        
        # need to do some more mask engineering on this term too
        mask = (Y > J) | (Y > L) | (L == 0)
        mask = np.tile(mask, (k_max, 1, k_max, 1, k_max, 1))
        # print(S3_yljh.shape)
        # print(mask.shape)
        # need to reshape
        
        S3_yljh[mask] = 0
        
        print(np.isnan(S3_yljh).mean())
        
        # marginalize over y
        self.A = (S1_iy*S2_kyljh*S3_yljh).sum(axis = 5)
        # print(self.A.shape)
          
    def b_array(self): 
        """
        array of coefficients for first term in the linear map
        
        probably wrong in various ways which need to be investigated further and tested
        
        need to check for zeros etc in these arrays
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


