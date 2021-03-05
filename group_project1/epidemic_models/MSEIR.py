import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def sir(a, b, N):
    return lambda t, y: np.array([-a/N*y[1]*y[0], a/N*y[1]*y[0]-b*y[1], b*y[1]])


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


def plot_trajectory(t, r0, s, i, e=None):
    plt.plot(s, i, lw=3, label='s, i progression')
    plt.plot([1/r0, 1/r0], [0, 1], '-.', lw=3, label='di/dt = 0')
    plt.plot(s[0], i[0], '.', ms=15, label='Initial Condition')
    plt.plot(s[-1], i[-1], '.', ms=15, label='Final Condition')
    plt.title('Disease State Trajectory')
    plt.xlabel('Susceptible')
    plt.ylabel('Infectious')
    plt.legend()
    plt.show()

