#!/bin/python

import numpy as np
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



if __name__ == "__main__":
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
    PB = 45 
    OM = 1
    I = 0 
    T0 = 20

    N = 100

    # Begin to process the data
    t = np.linspace(1, N, N)
    #t = (t - t[0]) * day # convert time from MJD to relative seconds

    x = keplerian_curve(t, alpha, delta, A, ECC, PB, OM, I, T0)
    x += np.random.randn(N) * np.sqrt(10**logW)
    #x += red_noise(logA, gamma, t)

    with open("test.tim", "w") as file:
        for i in range(N):
            file.write(f"{t[i]} {x[i]} {10**logW}\n")

    trues = np.array([alpha, delta, A, ECC, PB, OM, T0, logW])
    np.savetxt("trues.dat", trues)





