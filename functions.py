import numpy as np
from decimal import *

def build_adjacency_metropolis(N, G):
    '''
    Builds a combination matrix using a Metropolis rule.
    N: number of nodes.
    G: Adjacency matrix.
    '''
    A = np.zeros((N, N))
    nk = G.sum(axis=1)
    for k in range(N):
        for l in range(N):
            if G[k,l]==1 and k!=l:
                A[k,l] = 1/np.max([nk[k], nk[l]])
        A[k,k] = 1- A[k].sum()
    return A.T

def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p

def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix.
    mu: beliefs matrix.
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu
