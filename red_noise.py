#!/bin/python

import numpy as np
from scipy import linalg
from scipy.special import gamma as Gamma
from scipy.special import factorial
from scipy.optimize import curve_fit


day = 86400.0                   # Seconds per day
year =  31557600.0              # Seconds per year (yr = 365.25 days, so Julian years)


import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": "Helvetica",
#})


def save_image(filename):
	# PdfPages is a wrapper around pdf
	# file so there is no clash and create
	# files with no error.
    with PdfPages(filename) as p:

        # get_fignums Return list of existing
        # figure numbers
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        # iterating over the numbers in list
        for fig in figs:

            # and saving the files
            #fig.savefig(p, format='pdf', bbox_inches='tight', pad_inches = 0)
            fig.savefig(p, format='pdf', pad_inches = 0)
            plt.close(fig)

def fourierdesignmatrix(t, nmodes, Ttot=None):
    """
    Calculate the matrix of Fourier modes A, given a set of timestamps

    These are sine/cosine basis vectors at evenly separated frequency bins

    Mode 0: sin(f_0)
    Mode 1: cos(f_0)
    Mode 2: sin(f_1)
    ... etc

    :param nmodes:
        The number of modes that will be included (= 2*nfreq)
    :param Ttot:
        Total duration experiment (in case not given by t)

    :return:
        (A, freqs), with A the 'fourier design matrix', and f the associa

    """
    N = t.size
    A = np.zeros([N, nmodes])
    T = t.max() - t.min()

    if(nmodes % 2 != 0):
      print ("WARNING: Number of modes should be even!")

    if Ttot is None:
        deltaf = 1.0 / T
    else:
        deltaf = 1.0 / Ttot


    freqs1 = np.linspace(deltaf, (nmodes/2)*deltaf, nmodes//2)
    freqs = np.array([freqs1, freqs1]).T.flatten()


    for i in range(0, nmodes, 2):
        omega = 2 * np.pi * freqs[i]

        A[:,i] = np.cos(omega * t)
        A[:,i+1] = np.sin(omega * t)



    return (A, freqs)


def PSD (A, gamma, freqs):
    idx = np.where(np.abs(freqs) > 0)[0]

    res = np.zeros_like(freqs)
    res[idx] = A * (np.abs(freqs[idx]) ** (-gamma))

    return res 


def PSD_C (A, gamma, freqs, Ttot, f_L):

    n = len(freqs)//2
    jj = np.empty_like(freqs)

    for j in range(n):
        jj[2*j]   = (j+1)
        jj[2*j+1] = (j+1)


    idx = np.where(np.abs(freqs) > 0)[0]

    res = np.ones_like(freqs)

    res *= 4 * A * T**2 / (3-gamma) / jj**(gamma)

    return res * (-1)**(jj+1)


def cormat(A, gamma, tau_in, f_L):
    series = np.zeros_like(tau_in)
    term = np.zeros_like(tau_in)

    tau = f_L * tau_in * np.pi * 2

    gamma += 0

    n = 0
    term = 1/(1-gamma)
    series += term

    while(np.any(np.abs(term) > 1e-16)):
        n += 1

        term = (-1)**n * tau**(2*n) / factorial(2*n) / (2*n + 1 - gamma) 
        series += term

    a = Gamma(1 - gamma) * np.sin(np.pi * gamma / 2) * (tau ** (gamma - 1))

    return A / (f_L ** (gamma - 1)) * (a - series) * 2



def red_noise(logA, gamma, t):

    nmodes = len(t)
    T = t[-1] - t[0]

    # fing desirable covariance matrix for the noise
    F, freqs = fourierdesignmatrix(t, nmodes)
    phi = PSD(10**logA, gamma, freqs) / T * 2

    white = np.random.randn(len(phi))
    white *= np.sqrt(np.abs(phi))
    red = F @ white 

    return red

def correlogram(x):
    N = len(x)

    c = np.zeros_like(x)

    for m in range(N):
        c[m] = np.sum(x[0:N-m-1] * x[m:N-1]) / (N-m)

    return c

def periodogram(x, dt):
    rf = np.fft.fft(x)
    N = len(x)

    return (rf * rf.conj()).real * dt / N

def fit(ff, per):

    def f(ff, A, gamma):
        idx = np.where(np.abs(ff) > 0)[0]
        res = np.zeros_like(ff)
        res[idx] = A * (np.abs(ff[idx]) ** (-gamma))
        return res 

    p0 = [10**logA, gamma]

    popt, popv = curve_fit(f, ff, per, p0 = p0)

    print(p0)
    print(popt)

    return popt



if __name__ == "__main__":

    # set calculation parameters
    logA = -12
    gamma = 7/3

    t = np.linspace(0, 99, 100)
    nmodes = 200

    averaging_len = 1000

    # calculate etalon correlation matrix
    t1, t2 = np.meshgrid(t, t)
    tau = np.abs(t1 - t2) 
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    N = len(t)
    f_L = 1/T/100

    corr_etalon = cormat(10**logA, gamma, tau, f_L)

    eigenvalues, eigenvectors = linalg.eig(corr_etalon)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real



    # generate an ensemble of red noise realisations
    # via design matrix and find its correlation matrix

    F, freqs = fourierdesignmatrix(t, nmodes)
    phi = PSD(10**logA, gamma, freqs) / T * 2
    #phi = PSD_C(10**logA, gamma, freqs, T, f_L) 



    corr = np.zeros(shape = (N, N))
    corr = F @ np.diag(phi) @ F.T 

    #print(np.linalg.cond(corr_etalon), np.linalg.cond(corr[N//2:, :N//2]))


    C_mean = np.zeros(shape = (N, N))
    per = np.zeros_like(t)
    correl = np.zeros_like(t)
    ff = np.fft.fftfreq(N, dt)
    
    spec = np.zeros_like(ff)
    spec[ff != 0.0] = (10**logA) * np.abs(ff[ff != 0.0])**(-gamma)



    for i in range(averaging_len):
        ######## 1
        #red = np.random.randn(N)
        #rf = np.fft.fft(red)
        #rf = rf * np.sqrt(spec)
        #red = np.fft.ifft(rf).real


        ############### 2
        white = np.random.randn(len(phi))
        white *= np.sqrt(np.abs(phi))
        red = F @ white 


        ############## 3
        #w = np.random.randn(len(t))
        #w = w * np.sqrt(np.abs(eigenvalues))

        #red = np.matmul(eigenvectors, w)
        #red = np.random.multivariate_normal(np.zeros(N), corr)



        C_mean += np.tensordot(red, red, axes = 0) / averaging_len
        per += periodogram(red, dt) / averaging_len
        correl += correlogram(red)/averaging_len 

    fit(ff, per)

    plt.figure()
    plt.loglog(ff[ff>0], per[ff>0]/spec[ff>0], label = "$P(f) / S(f)$")
    plt.legend()



    fig, (axs1, axs2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    axs1.loglog(np.abs(ff), per, label = "Periodogramm")
    axs1.loglog(np.abs(ff), spec, "o", label = "Spectrum", markersize = 4)
    axs1.loglog(freqs, np.abs(phi), label = r"$\varphi$")

    axs1.legend()
    #axs1.set_ylim(1e-31, 1e-25)

    axs2.plot(tau[0, :], correl - correl[0], label = "$C_{mean}$")
    axs2.plot(tau[0, :], corr_etalon[0] - corr_etalon[0, 0], label = r"$C_{etalon}$")
    axs2.plot(tau[0, :], (corr[0] -corr[0, 0]), label = "$C_{phi}$")

    axs2.set_ylim(corr_etalon[0, :].min() - corr_etalon[0,0], 0)
    axs2.legend()




    plt.matshow(corr_etalon, interpolation = "nearest")
    plt.title(r"$C_{ij}\ (A = 10^{-14}, \gamma = -7/3)$")
    plt.colorbar()

    plt.matshow(C_mean, interpolation = "nearest")
    plt.title(r"$C_{mean}\ (A = 10^{-14}, \gamma = -7/3)$")
    plt.colorbar()

    plt.matshow(corr[N//2:, :N//2], interpolation = "nearest")
    plt.title(r"$F\phi F^T\ (A = 10^{-14}, \gamma = -7/3)$")
    plt.colorbar()



    save_image("123.pdf")







