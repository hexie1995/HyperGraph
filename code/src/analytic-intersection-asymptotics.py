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
        attempts to implement alpha from the writeup
        """
        
        k_max = self.k_max
        eta   = self.pars["eta"]
        BETA  = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        I = np.arange(k_max)[:, None, None, None, None, None]
        K = np.arange(k_max)[None, :, None, None, None, None]
        L = np.arange(k_max)[None, None, :, None, None, None]
        J = np.arange(k_max)[None, None, None, :, None, None]
        H = np.arange(k_max)[None, None, None, None, :, None]
        Y = np.arange(k_max)[None, None, None, None, None, :]
        
        # first term
        S1_iy = np.zeros((k_max, k_max))
        for i, y in product(range(k_max), range(k_max)):
            S1_iy[i, y] = sum(BETA[x]*GAMMA[i - y - x] for x in range(0, i - y + 1))
        
        # this would be a good spot to look for normalization shenanigans.
        
        # reshape and tile along missing axes
        S1_iy = S1_iy[:, None, None, None, None, :]
        S1_iy = np.tile(S1_iy, (1, k_max, k_max, k_max, k_max, 1)) 
        

        # create mask to zero out elements where |e \cap f| > |e|. 
        mask = Y > I 
        mask = np.tile(mask, (1, k_max, k_max, k_max, k_max, 1))
        # S1_iy[mask] = 0
        
        # just for testing 
        sparsity = (S1_iy == 0).mean()
        nan = np.isnan(S1_iy).mean()
        
        # print("A components")
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
        self.S1_iy = S1_iy
        # print(f"{S1_iy.shape =}")
        
        # second term (might not need reshaping, but check)
        # THIS IS PROBABLY ANOTHER GOOD PLACE TO CHECK FOR ISSUES WITH THE INTERSECTION SIZES
        S2_kyljh = binom(H, K)*binom(L - H, Y - K)/binom(L, Y)
        # S2_kyljh = S2_kyljh[None, None, :, :, :, :]
        S2_kyljh = np.tile(S2_kyljh, (k_max,  1, 1, k_max, 1, 1))
        # print(f"{S2_kyljh.shape =}")
        
        # zero out elements where the binomial coefficients are invalid
        mask = (K > H) | (Y > L) | (Y - K > L - H) | (H > L) | (H > J) | (L == 0) | (J == 0)
        # print(f"{mask.shape =}")
        mask = np.tile(mask, (k_max,  1, 1, 1, 1, 1))
        S2_kyljh[mask] = 0
        
        # just for testing
        sparsity = (S2_kyljh == 0).mean()
        nan = np.isnan(S2_kyljh).mean()
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
        
        self.S2_kyljh = S2_kyljh
        
        # third term
        S3_yljh = binom(L-1, Y-1) * eta**(Y-1) * (1-eta)**(L-Y)
        S3_yljh = np.tile(S3_yljh, (k_max, k_max, 1, k_max, k_max, 1))
        # print(f"{S3_yljh.shape =}")
        
        # need to do some more mask engineering on this term too
        mask =  (Y > L) | (L == 0)
        # print(f"{mask.shape =}")
        mask = np.tile(mask, (k_max,  k_max,1, k_max, k_max, 1))
        S3_yljh[mask] = 0
        
        # just for testing
        sparsity = (S3_yljh == 0).mean()
        nan = np.isnan(S3_yljh).mean()
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
        self.S3_yljh = S3_yljh
        
        # marginalize over y
        self.A = (S1_iy*S2_kyljh*S3_yljh).sum(axis = 5)
        
        
        # print("A")
        sparsity = (self.A == 0).mean()
        nan = np.isnan(self.A).mean()
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
    
    def omega_array(self): 
        
        k_max = self.k_max
        
        eta   = self.pars["eta"]
        BETA  = self.pars["beta"]
        GAMMA = self.pars["gamma"]
        
        I = np.arange(k_max)[:, None, None, None, None, None, None]
        K = np.arange(k_max)[None, :, None, None, None, None, None]
        L = np.arange(k_max)[None, None, :, None, None, None, None]
        J = np.arange(k_max)[None, None, None, :, None, None, None]
        H = np.arange(k_max)[None, None, None, None, :, None, None]
        S = np.arange(k_max)[None, None, None, None, None, :, None]
        X = np.arange(k_max)[None, None, None, None, None, None, :]
        
        
        t3_isx = BETA[I - S - X]
        
        
        
       
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
        M = binom(J-1, K-1)*eta**(K-1)*(1-eta)**(J-K) 
        mask = (K > J) | (J == 0) 
        M[mask] = 0
        # self.M = M
        
        # second term: probability of edge of size i given intersection of size k
        
        N = np.zeros((k_max, k_max))
        for i, k in product(range(1, k_max), range(k_max)):
            N[i, k] = sum(
                BETA[ell]*GAMMA[i-k-ell]
                for ell in range(0, i - k + 1)
            )
        
        for i, k, j in product(range(k_max), range(k_max), range(k_max)):
            self.B[i,k,j] = M[k,j] * N[i,k]
        
        # print("B")
        sparsity = (self.B == 0).mean()
        nan = np.isnan(self.B).mean()
        
        # print(f"{sparsity = :.4f}, {nan = :.4f}")
        
    def linear_map(self, T): 
        
        S = np.zeros_like(T)
        
        # case 1: k = 0
        # this CREATES k = 0 cases in the output tensor S
        # it does so by *also* using h = 0 cases in the input tensor T
        # numerically, this term can play quite a dominant role in the 
        # spectral structure
        # this term seems reliable, in the sense that in combination with the third term it captures the density of very large intersections nearly exactly
        to_add = 1/2*(
            np.einsum("lj,ilj -> ij", T[:,:,0], self.A[:,0,:,:,0]) + 
            np.einsum("jl,ijl -> ij", T[:,:,0], self.A[:,0,:,:,0])
        )
        S[:,:,0] = to_add
        
        # case 2: k >= 1
        ## term 1
        # this CREATES intersections FROM other intersections, not from h = 0 cases. 
        to_add =  1/2*(
            np.einsum("ljh,ikljh -> ijk", T, self.A[:,1:,:,:,:]) + 
            np.einsum("jlh,ikjlh -> ijk", T, self.A[:,1:,:,:,:])
        )
        S[:,:,1:] += to_add
        
        ## term 2
        # this USES  h = 0 cases in the input tensor T
        # this term seems reliable, in the sense that it captures the 
        # density of very large intersections nearly exactly. 
        to_add = 1*(
            np.einsum("lj,ikj -> ijk", T[1:,  :, 0], self.B[:,1:,:]) +
            np.einsum("jl,ikj -> ijk", T[ :, 1:, 0], self.B[:,1:,:])
            )
        
        S[:,:,1:] += to_add
        
        ## term 3 
        # this USES h = 0 cases in the input tensor in order to create intersections of size k = 1 through the extant node addition mechanism. 
        
        # MIGHT be the case that we only need to implement an omega array, and don't need to do anything else in particular to A or B. 
        
        
        
        
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


