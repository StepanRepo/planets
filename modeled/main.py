#!/bin/python

import numpy as np
from matplotlib import pyplot as plt
import myplot

import pandas as pd

from PTMCMCSampler import PTMCMCSampler
from chainconsumer import Chain, ChainConsumer, Truth, ChainConfig
from red_noise import *

day = 86400.0                   # Seconds per day
year =  31557600.0              # Seconds per year (yr = 365.25 days, so Julian years)




def TAU (t, ECC, PB, T0):
    M = 2*np.pi/PB * (t - T0)

    E0 = M
    E = ECC * np.sin(E0) + M

    #while (np.any(np.logical_not(np.isclose(E0, E)))):
    for _ in range(3):
        E0 = E
        E = ECC * np.sin(E0) + M

    return 2 * np.arctan(np.sqrt((1+ECC)/(1-ECC)) * np.tan(E/2)) 

def keplerian_curve(t, alpha, delta, A, ECC, PB, OM, I, T0):

    u = TAU(t, ECC, PB, T0) + OM

    n = len(t)

    cu = np.cos(u)
    su = np.sin(u)
    ci = np.cos(I)
    si = np.sin(I)

    vec = np.empty((n, 3))
    vec[:, 0] = cu
    vec[:, 1] = su * ci
    vec[:, 2] = su * si


    r = A * (1 - ECC**2) / (1 + ECC * cu)

    LOS = np.empty(3)
    LOS[0] = - np.cos(delta) * np.cos(alpha)
    LOS[1] = - np.cos(delta) * np.sin(alpha)
    LOS[2] = - np.sin(delta)

    for i in range(n):
        r[i] = r[i] * np.dot(vec[i, :], LOS)

    return r

    
def log_like (p, t, x, F, freqs):
    A, ECC, PB, OM, I, T0, logW, logA, gamma = p

    res = x - keplerian_curve(t, alpha, delta, *p[0:6])

    n = len(t)
    T = t[-1] - t[0]

    phi = PSD(10**logA, gamma, freqs) / T * 2
    sigsq = 10**logW

    Sigma = (F.T / sigsq) @ F + np.diag(1/phi)
    try:
        Sigma_cho = linalg.cho_factor(Sigma)
        Sigma_inv = linalg.cho_solve(Sigma_cho, np.eye(len(phi)))
    except:
        return -1e300

    d = F.T/sigsq
    C_inv = np.eye(n)/sigsq - d.T @ Sigma_inv @ d


    logdet_N = n*np.log(sigsq)
    logdet_B = np.sum(np.log(phi))
    logdet_Sigma = 2*np.sum(np.log(np.diag(Sigma_cho[0])))
    logdet = logdet_Sigma + logdet_B + logdet_N

    loglike =-.5 * (res.T @ C_inv @ res + logdet + n*np.log(2*np.pi))


    if not np.isfinite(loglike):
        loglike = -1e300

    return loglike

def log_like_W (p, t, x, F, freqs):
    A, ECC, PB, OM, T0, logW, = p

    res = x - keplerian_curve(t, alpha, delta, A, ECC, PB, OM, 0, T0)
    n = len(x)

    sigsq = 10**logW
    logdet = n*np.log(sigsq)

    loglike =-.5 * (np.dot(res, res)/sigsq + logdet + n*np.log(2*np.pi))


    if not np.isfinite(loglike):
        loglike = -1e300

    return loglike

def log_prior(u, left, right):
    if (np.any(u < left) or np.any(u > right)):
        return -np.inf
    else:
        return 0.0


if __name__ == "__main__":
    log_like = log_like_W

    # Generate data: 
    # orbital motion + white noise + red noise
    # True parameters

    # noise parameters
    logW = -7
    logA = -9
    gamma = 4/3

    # l=planetary parameters
    alpha = 0
    delta = 0
    A = 5e-4 
    ECC = 0.1 
    PB = 10 
    OM = 1
    I = 0 
    T0 = 20

    N = 100


    # Begin to process the data
    nmodes = 20

    t = np.linspace(1, N, N)
    #t = (t - t[0]) * day # convert time from MJD to relative seconds

    x = keplerian_curve(t, alpha, delta, A, ECC, PB, OM, I, T0)
    #x += red_noise(logA, gamma, t)
    x += np.random.randn(len(t)) * np.sqrt(10**logW)


    T = t[-1] - t[0]
    dt = t[1] - t[0]

    # calculate fourier design matrix for red noise
    F, freqs = fourierdesignmatrix(t, nmodes)


    labels = ["A", "ECC", "PB", "OM", "T0",\
            r'$log W$']#, \
            #r'$\log A$', r'$\gamma$']
    trues = np.array([A, ECC, PB, OM, np.mod(T0, PB), logW])


    left = np.array([1e-4,   0, dt,      0,  0, -8])
    right = np.array([1e-2, .9, T/2, np.pi, 40, -4])

    print(f"L_true = {log_like(trues, t, x, F, freqs):.1f}")

    # PTMCMC SAMPLING

    ndim = len(trues)
    coef = 2.38 / np.sqrt(ndim)
    coef *= .1
    cov = coef**2 * np.eye(ndim)
    #cov = np.diag([1e-8, 1e-2, 5e2, 1e-1, 1e-1, 1e2, 5e-3, 5e-3, 5e-3])
    p0 = np.random.uniform(left, right, ndim)


    sampler = PTMCMCSampler.PTSampler\
            (ndim, log_like, log_prior, \
            np.copy(cov), outDir='./chains',
             loglargs = [t, x, F, freqs],\
                     logpargs=[left, right])


    sampler.sample(p0, 100_000, burn=10_000,\
            thin=10, covUpdate=10_000, isave = 10,\
            Tmin = 1, Tmax = 1e3, writeHotChains = True)



    # Load the chian into Chainconsumer

    
    chain_array = np.loadtxt('chains/chain_1.txt')[-5000:, :-3]
    chain = pd.DataFrame(chain_array, \
            columns = labels + ["log_posterior"])
    chain = Chain(samples = chain, \
            name = "Orbit estimation with PT")

    consumer = ChainConsumer().add_chain(chain)
    consumer.add_truth(Truth(location = \
            {labels[i]: trues[i] for i in range(ndim)}))


    fig = consumer.plotter.plot_walks(\
            convolve=None, figsize = None,\
            plot_weights = False)
    fig = consumer.plotter.plot()



    # plot data and its spectrum with the expected one
    ff = np.fft.fftfreq(N, dt)
    per = periodogram(x, dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    ax1.plot(t, x)
    ax2.loglog(np.abs(ff), per)
    ax2.loglog(ff[ff>0], (10**logA) * ff[ff>0]**(-gamma))





    





    myplot.save_image("123.pdf")





