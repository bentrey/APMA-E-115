import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def sir(y, t, alpha, beta):
    """SIR model for population fraction"""
    s, i, r = y
    dsdt = -alpha * i * s
    didt = alpha * i * s - beta * i
    drdt = beta * i
    return [dsdt, didt, drdt]


def sirSim(y0, t, alpha, beta, showPlot=True):
    sirSoln = odeint(sir, y0, t, args=(alpha, beta))
    s, i, r = sirSoln.T

    # plot results
    if showPlot:
        plt.plot(t, s, label="S")
        plt.plot(t, i, label="I")
        plt.plot(t, r, label="R")
        plt.legend()
        plt.title('SIR Model')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.show()

    return sirSoln


def seir(y, t, alpha, beta, gamma):
    s, e, i, r = y
    dsdt = -beta * s * i
    dedt = beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]


def seir_control(y, t, alpha, beta, gamma, c):
    s, e, i, r = y
    dsdt = -(1-c)*beta * s * i
    dedt = (1-c)*beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]


def seir_control_dynamic(y, t, alpha, beta, gamma, cmax, tstar):
    s, e, i, r = y
    sl = cmax/tstar
    c = min(cmax, sl * t)

    dsdt = -(1-c)*beta * s * i
    dedt = (1-c)*beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]

def mseir():
    return


def seirSim(y0, beta, alpha, gamma, t, c=None,
            showPlot=True, cmax=None, tstar=None, dynamic=False):
    if not c:
        seirSoln = odeint(seir, y0, t, args=(alpha, beta, gamma))
    elif dynamic:
        seirSoln = odeint(seir_control_dynamic, y0, t,
                          args=(alpha, beta, gamma, cmax, tstar))
    else:
        seirSoln = odeint(seir_control, y0, t, args=(alpha, beta, gamma, c))
    s, e, i, r = seirSoln.T

    # plot results
    if showPlot:
        plt.plot(t, s, label="S")
        plt.plot(t, e, label="E")
        plt.plot(t, i, label="I")
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

