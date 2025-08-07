import numpy as np
from dataclasses import dataclass


@dataclass
class Parameters:
    "Model parameters, currently not including parameters related to protein systhesis and VLP assembly"
    mumax: float = 0.03  # maximum specific growth rate of cells (h^-1)
    kd: float = 0.0008  # cell death constant pre-infection (h^-1)
    Ks: int = 3  # Monod constant (mM)
    kdinf: float = 0.02  # cell death constant post-infection (h^-1)
    alpha: float = 0.08  # infection efficiency factor for uninfected cells
    alphaprime: float = 0.04  # infection efficiency factor for infected cells
    theta: int = 10  # time allowed for re-infection
    Ys: float = 8.46e5  # yield of cells on glutamine (cells/mM glutamine)

    deltatau: int = 36  # number of hours a cell could potentially sustain being infected before lysis (h)
    taulow: int = 12  # the minimum time post-infection virus & protein synthesis starts (h)
    tauhigh: int = 20  # the maximum time post-infection virus & protein synthesis starts (h)
    vmax: int = 300  # number of viruses above which the rate of synthesis no longer varies logarithmically and becomes a constant
    jhigh: int = 20  # number of viruses above which the start time for virion or protein synthesis no longer vary as a linear function of the number of viruses
    kv: float = 20  # virus synthesis rate constant (pfu/cell/h)
    # kp = 2.87e7    # protin synthesis rate constant (molecules/cell/h)

    def fj(self, j: int):
        "function describing the dependence of virus synthesis on the number of effective viruses in the cells"
        if j == 0:
            return 0
        elif j <= self.vmax:
            fj = np.log10(j + 1)
        else:
            fj = np.log10(self.vmax)
        return fj

    def tau(self, j: int):
        """Onset time of virus (and protein) synthesis as a capped linear relationship with j"""
        if j <= self.jhigh:
            return self.tauhigh - round((self.tauhigh - self.taulow) / (self.jhigh - 1) * (j - 1))
        else:
            return self.taulow
