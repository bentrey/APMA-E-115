import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def sir(a, b, N):
    return lambda t, y : np.array([-a/N*y[1]*y[0], a/N*y[1]*y[0]-b*y[1], b*y[1]])


def seir(beta, e, gamma):
    """
    :param y0: list contains initial values of S, I, E, R
    :return: seir function
    """
    S = y[0]
    I = y[1]
    E = y[2]
    R = y[3]

    return lambda t, y: np.array([-1*beta*y[0]*y[1], beta*y[0]*y[1]-e*y[2], e*y[2]-gamma*y[1],
                                   gamma*y[1]])


def seir(y, t, alpha, beta, gamma):
    s, e, i, r = y
    dsdt = -beta * s * i
    dedt = beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]


def mseir():
    return


def seirSim(y0, beta, alpha, gamma, t, showPlot=True):
    seirSoln = odeint(seir, y0, t, args=(alpha, beta, gamma))
    s, e, i, r = seirSoln.T

    # plot results
    if showPlot:
        plt.plot(t, s, label="S")
        plt.plot(t, e, label="I")
        plt.plot(t, i, label="E")
        plt.plot(t, r, label="R")
        plt.legend()
        plt.title('SEIR Model')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.show()

    return seirSoln


