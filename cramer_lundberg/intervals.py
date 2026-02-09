from scipy import stats
import numpy as np

def wilson_ci(psi_hat, N, confidence=0.995):
    z = stats.norm.ppf((1 + confidence) / 2)  # z = 2.807
    denominator = 1 + z**2 / N
    center = psi_hat + z**2 / (2*N)
    margin = z * np.sqrt(psi_hat*(1-psi_hat)/N + z**2/(4*N**2))
    
    L = (center - margin) / denominator
    U = (center + margin) / denominator
    return L, U
