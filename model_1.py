#!venv/bin/python

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


    
def log_like (p, t, x, errs, M):
    A, ECC, PB, OM, T0, EFAC, EQUAD = p

    res = x - keplerian_curve(t, alpha, delta, A, ECC, PB, OM, 0, T0)

    n = len(t)
    T = t[-1] - t[0]

    sigsq = (10**EFAC) * errs**2 + (10**EQUAD)**2

    T = M
    B_inv = np.zeros(T.shape[1])

    Sigma = (T.T / sigsq) @ T + np.diag(B_inv)
    try:
        Sigma_cho = linalg.cho_factor(Sigma)
        Sigma_inv = linalg.cho_solve(Sigma_cho, np.eye(T.shape[1]))
    except:
        return -1e300

    d = T.T/sigsq
    C_inv = np.diag(1/sigsq) - d.T @ Sigma_inv @ d


    logdet_N = np.sum(np.log(sigsq))
    logdet_B = 0
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
    every_chain = 10

    nsample = 30_000
    burnin = 10_000

    nmodes = 20

    t = psr.toas()[::every]
    x = psr.residuals()[::every]
    errs = psr.toaerrs[::every] * 1e-6
    N = psr.stoas / every

    alpha = psr["RAJ"].val
    delta = psr["DECJ"].val

    i = np.argsort(t)
    t, x, errs = t[i], x[i], errs[i]


    T = t[-1] - t[0]
    dt = t[1] - t[0]

    # calculate fourier design matrix for red noise
    #F, freqs = fourierdesignmatrix(t, nmodes)
    M = psr.designmatrix()[::every, :]

    M = M[i, :]
    for k in range(len(M[0, :])):
        M[:, k] /= np.max(np.abs(M[:, k]))


    labels = ["A", "ECC", "PB", "OM", "T0", r'log EFAC', r'log EQUAD']


    left = np.array([1e-3,   0, dt,      0,  t[0] + 1000, -1, -10], dtype = np.float64)
    right = np.array([1e-1, .9, T/2, np.pi, t[0] + 5000, 1, -1], dtype = np.float64)

    # PTMCMC SAMPLING

    ndim = len(labels)
    coef = 2.38 / np.sqrt(ndim)
    coef *= .2
    cov = coef**2 * np.eye(ndim)
    p0 = np.random.uniform(left, right, ndim)


    sampler = PTMCMCSampler.PTSampler\
            (ndim, log_like, log_prior, \
            np.copy(cov), outDir='./chains0',
             loglargs = [t, x, errs, M],\
                     logpargs=[left, right])


    sampler.sample(p0, 80_000, burn=10_000,\
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

    
    chain_array = np.loadtxt('chains0/chain_1.txt')[burnin::every_chain, :-3]
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
    if nsample/every_chain < 15_000:
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





