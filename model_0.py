#!/bin/python

import numpy as np
from matplotlib import pyplot as plt
import myplot

import pandas as pd

from PTMCMCSampler import PTMCMCSampler
from chainconsumer import Chain, ChainConsumer, Truth, ChainConfig
from red_noise import *


from glob import glob
import libstempo as T

day = 86400.0                   # Seconds per day
year =  31557600.0              # Seconds per year (yr = 365.25 days, so Julian years)


    
def log_like (p, t, x, errs, M, F, freqs):
    EFAC, EQUAD, logA, gamma = p

    res = x

    n = len(t)
    T = t[-1] - t[0]

    phi = PSD(10**logA, gamma, freqs) / T * 2
    sigsq = (10**EFAC) * errs**2 + (10**EQUAD)**2

    T = np.concatenate([M, F], axis = 1)
    B_inv = np.zeros(T.shape[1])
    B_inv[M.shape[1]:] = 1/phi

    Sigma = (T.T / sigsq) @ T + np.diag(B_inv)
    try:
        Sigma_cho = linalg.cho_factor(Sigma)
        Sigma_inv = linalg.cho_solve(Sigma_cho, np.eye(T.shape[1]))
    except:
        return -1e300

    d = T.T/sigsq
    C_inv = np.diag(1/sigsq) - d.T @ Sigma_inv @ d


    logdet_N = np.sum(np.log(sigsq))
    logdet_B = np.sum(np.log(phi))
    logdet_Sigma = 2*np.sum(np.log(np.diag(Sigma_cho[0])))
    logdet = logdet_Sigma + logdet_B + logdet_N

    loglike =-.5 * (res.T @ C_inv @ res + logdet + n*np.log(2*np.pi))


    if not np.isfinite(loglike):
        loglike = -1e300

    return loglike


def log_prior(u, left, right):
    if (np.any(u < left) or np.any(u > right)):
        return -np.inf
    else:
        return 0.0


if __name__ == "__main__":

    psr = T.tempopulsar(parfile = "./pulsars/0329+54/0329.par",
                        timfile = "./pulsars/0329+54/0329.tim")

    print("Psr was readed")

    every = 10
    nmodes = 20
    burnin = 2000

    t = psr.toas()[::every]
    x = psr.residuals()[::every]
    errs = psr.toaerrs[::every] * 1e-6
    N = psr.stoas / every







    T = t[-1] - t[0]
    dt = t[1] - t[0]

    # calculate fourier design matrix for red noise
    F, freqs = fourierdesignmatrix(t, nmodes)
    M = psr.designmatrix()[::every, :]


    labels = [r'log EFAC', r'log EQUAD', r'$\log A$', r'$\gamma$']


    left = np.array([-1, -10, -20, 0])
    right = np.array([1, -1,  -10, 7])


    # PTMCMC SAMPLING

    ndim = len(labels)
    coef = 2.38 / np.sqrt(ndim)
    coef *= .1
    cov = coef**2 * np.eye(ndim)
    p0 = np.random.uniform(left, right, ndim)


    sampler = PTMCMCSampler.PTSampler\
            (ndim, log_like, log_prior, \
            np.copy(cov), outDir='./chains0',
             loglargs = [t, x, errs, M, F, freqs],\
                     logpargs=[left, right])


    sampler.sample(p0, 50_000, burn=10_000,\
            thin=10, covUpdate=10_000, isave = 10,\
            Tmin = 1, Tmax = 1e8, writeHotChains = True)



## calculate model evidence
#    files = glob("chains0/chain*")
#    betas = np.empty(len(files)+1)
#    lls = np.empty_like(betas)
#
#    betas[0] = 0
#    lls[0] = 0
#
#    for i, file in enumerate(files):
#        temp = file.split("_")[1].split(".")
#        temp = np.float64(f"{temp[0]}.{temp[1]}")
#
#        chain = np.loadtxt(file)[burnin:, -3]
#
#        betas[i+1] = 1/temp
#        lls[i+1] = np.mean(chain)
#
#
#    i = np.argsort(betas)
#
#    betas = betas[i]
#    lls = lls[i]
#
#
#    logz = 0
#
#    for i in range(len(lls)-1):
#        logz += (lls[i] + lls[i+1]) * (betas[i+1] - betas[i]) / 2 



    # Load the chian into Chainconsumer

    
    chain_array = np.loadtxt('chains0/chain_1.txt')[burnin:, :-3]
    chain = pd.DataFrame(chain_array, \
            columns = labels + ["log_posterior"])
    chain = Chain(samples = chain, \
            name = "Orbit estimation with PT")

    consumer = ChainConsumer().add_chain(chain)



    # plot data and its spectrum with the expected one
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(t, x)
    ax1.set_title("TRs")
    ax1.set_xlabel("time, [days]")
    ax1.set_ylabel("TR, [sec]")


    # plot chains traces
    fig = consumer.plotter.plot_walks(\
            convolve=None, figsize = None,\
            plot_weights = False)

    # plot triangle plot
    fig = consumer.plotter.plot()


#    # plot temperature ledder
#    fig, ax = plt.subplots(1, 1)
#    fig.tight_layout(pad = 5)
#
#    ax.plot(betas[1:], lls[1:], "o")
#    ax.plot(betas[1:], lls[1:], label = f"$\log Z = {logz:.1f}$")
#    ax.set_xscale("log")
#
#    ax.set_title(r"Evidence estimation")
#    ax.set_xlabel(r"$\beta$")
#    ax.set_ylabel(r"$\langle \log L^{\beta} \rangle$")
#    ax.legend()







    myplot.save_image("123.pdf")





