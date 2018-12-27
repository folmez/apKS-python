import numpy as np
from scipy import stats

def my_lognorm_pdf(x, mu, sigma):
    shape   = sigma
    loc     = 0
    scale   = np.exp(mu)
    return stats.lognorm.pdf(x, shape, loc, scale)

def my_lognorm_cdf(x, mu, sigma):
    shape   = sigma
    loc     = 0
    scale   = np.exp(mu)
    return stats.lognorm.cdf(x, shape, loc, scale)

def my_lognorm_inv_cdf(q, mu, sigma):
    shape   = sigma
    loc     = 0
    scale   = np.exp(mu)
    return stats.lognorm.ppf(q, shape, loc, scale)
