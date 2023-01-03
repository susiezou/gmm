#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PLOT_NORMAL.PY
Date: Wednesday, October 26 2011
Description: Visualization of the normal distribution.
"""

import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
import pylab as pl
import matplotlib
from matplotlib.ticker import NullFormatter
from matplotlib.widgets import Slider
import pdb
from normal import Normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

def draw2dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 2d normal pdf.
    """
    # create a meshgrid centered at mu that takes into account the variance in x and y
    delta = 0.01

    lower_xlim = norm.mu[0] - (3.0 * norm.E[0,0] ** 0.5)
    upper_xlim = norm.mu[0] + (3.0 * norm.E[0,0] ** 0.5)
    lower_ylim = norm.mu[1] - (3.0 * norm.E[1,1] ** 0.5)
    upper_ylim = norm.mu[1] + (3.0 * norm.E[1,1] ** 0.5)

    x = np.arange(lower_xlim, upper_xlim, delta)
    y = np.arange(lower_ylim, upper_ylim, delta)

    X,Y = np.meshgrid(x,y)

    # remember sqrts!
    Z = bivariate_normal(X, Y, sigmax=np.sqrt(norm.E[0,0]), sigmay=np.sqrt(norm.E[1,1]), mux=norm.mu[0], muy=norm.mu[1], sigmaxy=norm.E[0,1])

    minlim = min(lower_xlim, lower_ylim)
    maxlim = max(upper_xlim, upper_ylim)

    # Plot the normalized faithful data points.
    if not axes:
        fig = pl.figure( figsize=(4,4))
        pl.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)
    else:
        axes.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)

    if show:
        pl.show()


def draw3dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 3d normal pdf.
    """
    # create a meshgrid centered at mu that takes into account the variance in x and y
    delta = 0.01

    lower_xlim = norm.mu[0] - (2.0 * norm.E[0,0] ** 0.5)
    upper_xlim = norm.mu[0] + (2.0 * norm.E[0,0] ** 0.5)
    lower_ylim = norm.mu[1] - (2.0 * norm.E[1,1] ** 0.5)
    upper_ylim = norm.mu[1] + (2.0 * norm.E[1,1] ** 0.5)

    x = np.arange(lower_xlim, upper_xlim, delta)
    y = np.arange(lower_ylim, upper_ylim, delta)
    X, Y = np.meshgrid(x,y)
    e, xx, yy = confidence_ellipse(norm.mu, norm.E)
    # remember sqrts!
    # Z = bivariate_normal(X, Y, sigmax=np.sqrt(norm.E[0,0]), sigmay=np.sqrt(norm.E[1,1]), mux=norm.mu[0], muy=norm.mu[1], sigmaxy=norm.E[0,1])
    d = norm.A_matrix @ np.array([xx, yy]) + norm.b
    dd = norm.A_matrix @ np.array([X.ravel(), Y.ravel()]) + norm.b
    # Plot the normalized faithful data points.
    if not axes:
        fig = pl.figure( figsize=(4,4))
        axes = pl.axes(projection='3d')
        axes.contour(xx,yy,np.asarray(d).ravel())
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)
    else:
        # axes.plot_surface(X, Y, np.array(dd).reshape(X.shape), cmap='jet')
        axes.plot(xx,yy,np.array(d).ravel())
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)

    if show:
        pl.show()


def evalpdf(norm):
    delta = 0.025
    mu = norm.mu[0]
    sigma = norm.E[0,0]
    lower_xlim = mu - (2.0 * sigma)
    upper_xlim = mu + (2.0 * sigma)
    x = np.arange(lower_xlim,upper_xlim, delta)
    y = matplotlib.mlab.normpdf(x, mu, np.sqrt(sigma))
    return x,y

def draw1dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 1d normal pdf. Used for plotting the conditionals in simple test cases.
    """
    x,y = evalpdf(norm)
    if axes is None:
        pl.plot(x,y)
    else:
        return axes.plot(y,x)

    if show:
        pl.show()

def draw2d1dnormal(norm, cnorm, show = False):

    pl.figure(1, figsize=(8,8))

    nullfmt = NullFormatter()

    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    draw2dnormal(norm, axes = ax2d)
    draw1dnormal(cnorm, axes = ax1d)
    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    ax2d.plot(x,y)


def draw_slider_demo(norm):

    fig = pl.figure(1, figsize=(8,8))
        
    nullfmt = NullFormatter()

    cnorm = norm.condition([0],2.0)

    rect_slide = [0.1, 0.85, 0.65 + 0.1, 0.05]
    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    axslide = pl.axes(rect_slide)
    slider = Slider(axslide, 'Cond', -4.0,4.0,valinit=2.0)
        
    draw2dnormal(norm, axes = ax2d)
    l2, = draw1dnormal(cnorm, axes = ax1d)

    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    l1, = ax2d.plot(x,y)
    
    def update(val):
        cnorm = norm.condition([0],val)
        x = [cnorm.cond['data'], cnorm.cond['data']]
        l1.set_xdata(x)
        x,y = evalpdf(cnorm)
        print(cnorm)
        #print y
        l2.set_xdata(y)
        l2.set_ydata(x)
        pl.draw()
            

    slider.on_changed(update)
    
    return slider


def confidence_ellipse( mu, cov, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((mu[0], mu[1]), width=ell_radius_x * 2 * scale_x, height=ell_radius_y * 2 * scale_y,
                      facecolor=facecolor, **kwargs)



    # transf = transforms.Affine2D() \
    #     .rotate_deg(45) \
    #     .scale(scale_x, scale_y)  # \
    #     # .translate(mu[0], mu[1])
    #
    # ellipse.set_transform(transf + ax.transData)

    # Get the path
    path = ellipse.get_path()
    # Get the list of path vertices

    vertices = path.vertices.copy()
    # Transform the vertices so that they have the correct coordinates
    vertices = ellipse.get_patch_transform().transform(vertices)
    xx, yy = vertices.T
    return ellipse, xx, yy #ax.add_patch(ellipse)


if __name__ == '__main__':
    # Tests for the ConditionalNormal class...
    mu = [1.5, 0.5]
    sigma = [[1.0, 0.5], [0.5, 1.0]]
    n = Normal(2, mu = mu, sigma = sigma)
    sl = draw_slider_demo(n)
    pl.show()
