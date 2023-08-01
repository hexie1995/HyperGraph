import unittest

import likelihoods as ll 
import em_new as em
import model
import numpy as np
import scipy.special as ss


class TestLikelihoods(unittest.TestCase):
    
    def test_guaranteed_edge_sample_likelihood(self):
        t = np.array([6, 2, 3])
        x = np.array([3, 2, 4])
        eta = 0.4

        p = ll.edge_sample_likelihood(x, t, eta = eta)
        
        expected = (eta**(x-1))*((1-eta)**(t-x))
        expected[x > t] = 0
        
        self.assertAlmostEqual(expected[0], p[0], msg = "edge sample likelihood returns expected result when number of successes is less than number of possibilities")
        
        self.assertAlmostEqual(expected[1], p[1], msg = "edge sample likelihood returns expected result when number of successes is equal to number of possibilities")
        
        self.assertAlmostEqual(expected[2], p[2], msg = "edge sample likelihood returns expected result (0) when number of successes exceeds number of possibilities")

    def test_novel_nodes_likelihood(self):
        beta = 0.2
        ell = np.array([4, 0, -1])
        p   = ll.novel_nodes_likelihood(ell, beta = beta)
        
        expected = np.exp(-beta)*beta**ell / ss.factorial(ell)
        expected[ell < 0] = 0
        
        self.assertAlmostEqual(expected[0], p[0], msg = "expected result when adding a positive number of novel nodes")
        self.assertAlmostEqual(expected[1], p[1], msg = "expected result when adding no novel nodes")
        self.assertAlmostEqual(expected[2], p[2], msg = "expected result (0) when adding a negative number of novel nodes")
        
    def test_nodes_from_hypergraph_likelihood(self):
        gamma = 0.5
        ell = np.array([4, 0, -1])
        p   = ll.nodes_from_hypergraph_likelihood(ell, gamma = gamma)
        
        expected = np.exp(-gamma)*gamma**ell / ss.factorial(ell)
        expected[ell < 0] = 0
        
        self.assertAlmostEqual(expected[0], p[0], msg = "expected result when adding a positive number of novel nodes")
        self.assertAlmostEqual(expected[1], p[1], msg = "expected result when adding no novel nodes")
        self.assertAlmostEqual(expected[2], p[2], msg = "expected result (0) when adding a negative number of novel nodes")
        
class TestEM(unittest.TestCase):
    
    def setUp(self):
        # generate some sample data

        self.eta   = 0.5   # node retention in edges
        self.beta  = 0.5   # poisson number of novel nodes
        self.gamma = 0.5   # poisson number of nodes from graph

        timesteps = int(1e2)

        self.H = model.GrowingHypergraph()
        self.H.add_edge((0, 1))
        self.H.add_edge((2, 3))

        for _ in range(timesteps):
            self.H.sample_edge(self.eta, self.gamma, self.beta, True)

        self.em = em.EM(self.H)
        self.em.E_step()
        
    def test_conformable_shapes(self):
        self.assertEqual(self.em.intersection_array.shape, self.em.CHI.shape, msg = "array of edge intersection data and self.CHI have the same shape")
        
    def test_CHI_normalization(self):
        self.assertAlmostEqual(self.em.CHI.sum(axis = (1, 2, 3))[78], 1.0, msg = "the entry of self.CHI for a single edge sums to 1")

    def test_CHI_inequalities(self):
        CHI = self.em.CHI
        
        edge_sizes = self.H.edge_size_sequence()
        k_max = edge_sizes.max()
        k = np.arange(0, k_max + 1)
        
        I, J, K, L = np.meshgrid(k, k, k, k, indexing = "ij")
        
        mask = (K > J)[edge_sizes, :, :, :]
        
        self.assertAlmostEqual(np.nansum(CHI[mask]), 0.0, msg = "Probability mass on events where size of intersection K exceeds size of edge J must be 0")
        
        mask = (K > I)[edge_sizes, :, :, :]
        
        self.assertAlmostEqual(np.nansum(CHI[mask]), 0.0, msg = "Probability mass on events where size of intersection K exceeds size of edge J must be 0")
        
        mask = (L < 0)[edge_sizes,:,:,:]
        
        self.assertAlmostEqual(np.nansum(CHI[mask]), 0.0, msg = "Probability mass on adding a negative number of novel nodes must be 0")
        
        mask = (I - K - L < 0)[edge_sizes,:,:,:]
        
        self.assertAlmostEqual(np.nansum(CHI[mask]), 0.0, msg = "Probability mass on adding a negative number of nodes from hypergraph must be 0")
    
    def test_increase_in_marginal_log_likelihood(self):
        """
        should test whether a single round of EM increases the marginal log-likelihood
        
        need to correctly calculate the marginal first
        """
        pass

    
    def test_marginal_log_likelihood(self):
        
        ll = self.em.marginal_log_likelihood()
        
        self.assertTrue(isinstance(ll, float), msg = "marginal log-likelihood can be computed without runtime error")
    
    def test_fit(self):
        
        self.em.fit()
        self.assertTrue(self.em.ll_history[-1] >= self.em.ll_history[0], msg = "ll at end of EM is larger than at beginning")
         
    
if __name__ == '__main__':
    unittest.main()
