import numpy as np
from scipy.special import gammaln


def pinf(j: int, alpha: float, V1: float, N: float):
    """Poisson probability of infection.
    Args:
        j: number of viruses infecting a cell
        alpha: infection e$ciency factor
        V1: extracellular virus concentration
        N: cell concentration
    """
    # no virus or no cells to infect
    if j == 0 or V1 == 0 or N == 0:
        return 0
    else:
        dynMOI = alpha * V1 / N
        return np.exp(-dynMOI + j*np.log(dynMOI) - gammaln(j + 1))

def jmin_jmax(alpha: float, V1: float, N: float):
    """Upper and lower bound of infecting virus number j.
    Args:
        alpha:
        V1:
        N:
    """
    # extreme cases: no virus or no cells
    if V1 == 0 or int(N) == 0:
        return 0, 0
    
    dynmoi = alpha * V1 / N
    if dynmoi < 1:
        return 0, 4
    elif dynmoi < 10:
        return 0, round(dynmoi+1)*2
    elif dynmoi < 60:
        return max(0, round(dynmoi - 20)), round(dynmoi + 20)
    elif dynmoi < 500:
        rad = 0.1*dynmoi + 16
        return round(dynmoi - rad), round(dynmoi + rad)
    else:
        return 440, 560

def td(V1: float, No: float):
    """Parameter t_d is the time after infection when cell viability 
    starts to drop quickly. t_d is a function of initial MOI.
    Args:
        V1:
        No:
    """
    MOI = V1 / No
    if MOI < 1:
        return 60
    elif MOI <= 20:
        return 48 * np.exp(-0.06 * MOI)
    elif MOI > 20:
        return 14
