#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: gmm.PY
Date: Friday, June 24 2011/Volumes/NO NAME/seds/nodes/gmm.py
Description: A python class for creating and manipulating GMMs.
"""

import scipy.cluster.vq as vq
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
npa = np.array

import sys; sys.path.append('.')
import pdb

#import matplotlib
import pylab
from normal import Normal

class GMM(object):

    def __init__(self, dim = None, ncomps = None, data = None,  method = None, filename = None, params = None):

        if not filename is None:  # load from file
            self.load_model(filename)

        elif not params is None: # initialize with parameters directly
            self.comps = params['comps']
            self.ncomps = params['ncomps']
            self.dim = params['dim']
            self.priors = params['priors']

        elif not data is None: # initialize from data

            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps
            self.comps = []

            if method is "uniform":
                # uniformly assign data points to components then estimate the parameters
                npr.shuffle(data)
                n = len(data)
                s = n / ncomps
                for i in range(ncomps):
                    self.comps.append(Normal(dim, data = data[i * s: (i+1) * s]))

                self.priors = np.ones(ncomps, dtype = "double") / ncomps

            elif method is "random":
                # choose ncomp points from data randomly then estimate the parameters
                mus = pr.sample(data,ncomps)
                clusters = [[] for i in range(ncomps)]
                for d in data:
                    i = np.argmin([la.norm(d - m) for m in mus])
                    clusters[i].append(d)

                for i in range(ncomps):
                    print (mus[i], clusters[i])
                    self.comps.append(Normal(dim, mu = mus[i], sigma = np.cov(clusters[i], rowvar=0)))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            elif method is "kmeans":
                # use kmeans to initialize the parameters
                (centroids, labels) = vq.kmeans2(data, ncomps, minit="points", iter=100)
                clusters = [[] for i in range(ncomps)]
                for (l,d) in zip(labels,data):
                    clusters[l].append(d)

                # will end up recomputing the cluster centers
                for cluster in clusters:
                    self.comps.append(Normal(dim, data = cluster))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            else:
                raise ValueError("Unknown method type!")

        else:

            # these need to be defined
            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps

            self.comps = []

            for i in range(ncomps):
                self.comps.append(Normal(dim))

            self.priors = np.ones(ncomps, dtype='double') / ncomps

    def __str__(self):
        res = "%d" % self.dim
        res += "\n%s" % str(self.priors)
        for comp in self.comps:
            res += "\n%s" % str(comp)
        return res

    def save_model(self):
        pass

    def load_model(self):
        pass

    def mean(self):
        return np.sum([self.priors[i] * self.comps[i].mean() for i in range(self.ncomps)], axis=0)

    def covariance(self): # computed using Dan's method
        m = self.mean()
        s = -np.outer(m,m)

        for i in range(self.ncomps):
            cm = self.comps[i].mean()
            cvar = self.comps[i].covariance()
            s += self.priors[i] * (np.outer(cm,cm) + cvar)

        return s

    def pdf(self, x):
        responses = [comp.pdf(x) for comp in self.comps]
        return np.dot(self.priors, responses)

    def condition(self, indices, x):
        """
        Create a new gmm conditioned on data x at indices.
        """
        condition_comps = []
        marginal_comps = []

        for comp in self.comps:
            condition_comps.append(comp.condition(indices, x))
            marginal_comps.append(comp.marginalize(indices))

        new_priors = []
        for (i,prior) in enumerate(self.priors):
            new_priors.append(prior * marginal_comps[i].pdf(x))
        new_priors = npa(new_priors) / np.sum(new_priors)

        params = {'ncomps' : self.ncomps, 'comps' : condition_comps,
                  'priors' : new_priors, 'dim' : marginal_comps[0].dim}

        return GMM(params = params)

    def em(self, data, nsteps = 100, specify=False):

        k = self.ncomps
        d = self.dim
        n = len(data)
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)
        d_x = X.shape[1]

        for l in range(nsteps):

            # E step

            responses = np.zeros((k,n))


            for i in range(k):
                responses[i, :] = self.priors[i] * self.comps[i].pdf_sci(data)

            responses /= np.sum(responses, axis=0) # normalize the weights

            # M step

            N = np.sum(responses, axis=1)

            responses = np.mat(responses)
            for i in range(k):
                if specify:
                    mu = np.dot(responses[i, :], X) / N[i]
                    cov_k = np.multiply((X - mu).T, responses[i, :]) @ (X - mu) / N[i]
                    # update A matrix
                    X_ = np.multiply((X - mu).T, np.sqrt(responses[i, :]))
                    yk_ = np.dot(responses[i, :], Y) / N[i]
                    Y_ = np.multiply((Y - yk_).T, np.sqrt(responses[i, :]))
                    A = np.dot(Y_, np.linalg.pinv(X_))  # A[k] ~ 1 * D
                    # update b
                    b_ = Y - np.dot(X, A.T)
                    b = np.dot(responses[i, :], b_) / N[i]
                    # update sigma
                    sig = np.dot(np.multiply((b_ - b).T, responses[i, :]), (b_ - b)) / N[i]

                    cov_sig = np.zeros((d, d))
                    cov_sig[:d_x, :d_x] = cov_k
                    cov_sig[:d_x, d_x:] = cov_k @ A.T
                    cov_sig[d_x:, :d_x] = A @ cov_k
                    cov_sig[d_x:, d_x:] = sig + A @ cov_sig[:d_x, d_x:]
                    # update the normal with new parameters
                    self.comps[i].update(np.array(np.c_[mu, A @ mu.T + b]).ravel(), cov_sig)
                    self.comps[i].A_matrix = A
                    self.comps[i].b = b
                    self.comps[i].plane_noise = sig
                    # print("multi-dimension gmm with plane conditions")
                else:
                    mu = np.dot(responses[i, :], data) / N[i]
                    cov_k = np.multiply((data - mu).T, responses[i, :]) @ (data - mu) / N[i]
                    mu = np.array(mu).ravel()
                    cov_k = np.array(cov_k)
                    self.comps[i].update(mu, cov_k)
                    self.comps[i].A_matrix = cov_k[d_x:, :d_x] @  np.linalg.inv(cov_k[:d_x, :d_x])
                    self.comps[i].b = mu[d_x:] - self.comps[i].A_matrix @ mu[:d_x].reshape(-1, 1)
                    self.comps[i].plane_noise = cov_k[d_x:, d_x:] - self.comps[i].A_matrix @ cov_k[:d_x, d_x:]
                    # print("multi-dimension gmm")
                self.priors[i] = N[i] / np.sum(N) # normalize the new priors


def shownormal(data,gmm):

    xnorm = data[:,0]
    ynorm = data[:,1]

    # Plot the normalized faithful data points.
    fig = pylab.figure(num = 1, figsize=(4,4))
    axes = fig.add_subplot(111)
    axes.plot(xnorm,ynorm, '+')

    # Plot the ellipses representing the principle components of the normals.
    for comp in gmm.comps:
        comp.patch(axes)

    pylab.draw()
    pylab.show()


if __name__ == '__main__':

    """
    Tests for gmm module.
    """


    # x = npr.randn(20, 2)

    # print "No data"
    # gmm = gmm(2,1,2) # possibly also broken
    # print gmm

    # print "Uniform"
    # gmm = gmm(2,1,2,data = x, method = "uniform")
    # print gmm

    # print "Random"
    # gmm = gmm(2,1,2,data = x, method = "random") # broken
    # print gmm

    # print "Kmeans"
    # gmm = gmm(2,1,2,data = x, method = "kmeans") # possibly broken
    # print gmm


    x = np.arange(-10,30)
    #y = x ** 2 + npr.randn(20)
    y = x + npr.randn(40) # simple linear function
    #y = np.sin(x) + npr.randn(20)
    data = np.vstack([x,y]).T
    print(data.shape)


    gmm = GMM(dim = 2, ncomps = 4,data = data, method = "random")
    print(gmm)
    shownormal(data,gmm)

    gmm.em(data,nsteps=1000)
    shownormal(data,gmm)
    print(gmm)
    ngmm = gmm.condition([0],[-3])
    print(ngmm.mean())
    print(ngmm.covariance())
