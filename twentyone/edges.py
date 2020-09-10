import numpy as np
from twentyone.units import *


class EDGES_fit:
    def __init__(self, z, params="default"):

        if params is "default":
            A = np.array([0.5, 0.5 + 0.5, 0.5 - 0.2])
            nu_0 = np.array([78.0, 78.0 + 1.0, 78.0 - 1.0])
            w = np.array([19.0, 19.0 + 4, 19.0 - 2.0])
            tau = np.array([7.0, 7.0 + 5.0, 7.0 - 3.0])
        elif params == "MC":
            A = np.array([0.52, 0.52 + 0.42, 0.52 - 0.18])
            nu_0 = np.array([78.3, 78.3 + 0.2, 78.3 - 0.3])
            w = np.array([20.7, 20.7 + 0.8, 20.7 - 0.7])
            tau = np.array([6.5, 6.5 + 5.6, 6.5 - 2.5])
        else:
            A, tau, nu_0, w = params

        self.A = A
        self.tau = tau
        self.nu_0 = nu_0
        self.w = w

        opz_to_freq = lambda opz: ((omega_21 / (opz) / (1e6 * Hz) / (2 * np.pi)))

        self.nu = opz_to_freq(1 + z)

        self.T21_fit = self.T21(self.A[:, None], self.B(self.nu[None, :], self.nu_0[:, None], self.w[:, None], self.tau[:, None]), self.tau[:, None])

    def T21(self, A, B, tau):
        return -A * (1 - np.exp(-tau * np.exp(B))) / (1 - np.exp(-tau))

    def B(self, nu, nu_0, w, tau):
        return 4 * (nu - nu_0) ** 2 / w ** 2 * np.log(-np.log((1 + np.exp(-tau)) / 2.0) / tau)


def opz_to_freq(opz):
    return ((omega_21 / (opz) / (1e6 * Hz) / (2 * np.pi)))

def freq_to_opz(nu):
    return ((omega_21 / (2 * np.pi * (nu * 1e6 * Hz))))

