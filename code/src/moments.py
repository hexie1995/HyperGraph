import numpy as np
from scipy.optimize import fsolve
from itertools import product

def moments(H):
    """
    compute selected moments of the input GrowingHypergraph
    """
    k = H.edge_size_sequence()
    d = H.degree_sequence()
    
    k1 = k.mean()
    k2 = (k**2).mean() 
    d1 = d.mean()
    
    return np.array([d1, k1, k2])

def analytic_moments(theta):
    """
    estimate the selected moments of a GrowingHypergraph in terms of the 
    parameter vector theta. 
    
    theta = (eta, gamma, beta)
    """
    eta, gamma, beta = theta[0], theta[1], theta[2]
    
    k1 = 1 + (beta + gamma) / (1 - eta)
    
    d1 = 1/beta * k1
    
    k2  = beta*(1+beta) + gamma*(1+gamma)+2*beta*gamma 
    k2 += k1*(2*(beta + gamma) + eta*(1-eta))
    k2 /= 1-eta**2 
    
    return np.array([d1, k1, k2])


def is_valid(theta):
    return (0 <= theta[0] <= 1) and (0 <= theta[1]) and (0 <= theta[2])

def method_of_moments_estimator(H, theta0 = None, max_init = 100):

    M = moments(H)

    def f(theta):
        return analytic_moments(theta) - M
    
    if not theta0:
        for i in range(max_init):
            
            theta0 = np.random.rand(3)

        theta_hat = fsolve(f, theta0)
        obj = np.sqrt((f(theta_hat)**2).mean())
        
        if (obj < 1e-8) and is_valid(theta_hat):
            return theta_hat, obj

    print("No valid solution reached in {max_init} initializations")
    return np.array([-1, -1, -1]), np.inf
