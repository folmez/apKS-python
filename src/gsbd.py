"""
GSBD(X, alpha, xmin, xmax, X_dattype) generates bootstrap samples of
data with a semiparametric approach as described in section 4.1 of
Clauset et. al. 2009 ('Power-law distributions in empirical data'). The
approach can be summarized as follows: Let n be the length of the data
and n_pl be the number of data points in the fitted power-law region.
Generate a new data set one data point at a by
   (i) picking an element from non-power-law subset of the original
       data set with probability 1 - n_pl/n
   (ii) picking a power-law distributed data point (according to the
       power-law-fit) with probability n_pl/n
"""

import numpy as np

def gsbd(X, alpha, xmin, xmax, X_data_type):
    # Get total and non-power-law sample sizes
    n = len(X)
    X_non_pl = X[(X<xmin) | (xmax<X)] # non-power-law subset
    n_non_pl = len(X_non_pl)

    # (i) Choose elements from the non-power-law subset with prob. n_non_pl / n
    n1 = sum(np.random.rand(n) < n_non_pl / n)
    sbX1 = np.random.choice(X_non_pl, size=n1, replace=True)

    # (ii) Generate a power-law distributed sample of size n-n1
    n2 = n-n1
    if X_data_type == 'REAL':
        # Generate a continuous power-law sample in [xmin, xmax]
        sbX2 = gen_power_law_sample(alpha, xmin, xmax, n2)
    elif X_data_type == 'INTS':
        # Generate a continuous power-law sample in [xmin-0.5, xmax+0.5] and
        # then round it to an array of integers
        sbX2 = np.round(gen_power_law_sample(alpha, xmin-0.5, xmax+0.5, n))

    # Return the union of the two arrays
    return np.sort(np.append(sbX1, sbX2))

def gen_power_law_sample(a, xmin, xmax, n):
    U = np.random.rand(n)
    if not np.isclose(a, 1.0):
        b = 1.0-a
        return (U * (xmax**b-xmin**b) + xmin**b) ** (1.0/b)
    else:
        return np.exp( U * (np.log(xmax)-np.log(xmin)) + np.log(xmin) )
