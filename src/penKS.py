"""
penKS(X, X_dattype) calculates a bounder power-law fit to the given
data set X. The penalty slope is 0 by default, i.e. the default is the
KS method for bounded power law fitting.

Algorithm:
    1) Use pKS-detected or lower-bound and upper-bound
        over a subset of the sample, namely a
        logarithmically-equally-spaced subset coming from elspd.py
    2) Fix xmax from step (1) as pKS-detected-upper-bound
        and detect lower-bound over the sample
    3) Fix xmin detected from step (2) and
        detect upper-bound over the sample
    4) Loop between 2 and 3 until an equilibrium is
        determined

In summary, we first minimize the KS metric over a grid and then finetune the
bounds one at a time.
"""

import numpy as np
import src

def penKS(X, X_dattype, xmin_vec=0, xmax_vec=0, \
            pen_slope=0, data_title='Untitled data', \
            min_nr_trial_pts_in_a_decade=10, interval_length_threshold=10):

    # Define lower-bound and upper-bound candidates
    if xmin_vec is 0 and xmax_vec is 0:
        LmX = src.elspd(X, min_nr_trial_pts_in_a_decade)
        xmin_vec, xmax_vec = LmX, LmX

    # Get the sizes of candidate arrays
    nr_xmins, nr_xmaxs = len(xmin_vec), len(xmax_vec)

    # Initialize KS and a power-law exponent matrix
    KS_mat = np.zeros([nr_xmins, nr_xmaxs])
    alpha_mat = np.zeros([nr_xmins, nr_xmaxs])

    # Calculate the exponent and KS metric for each [xmin, xmax] interval
    for i, xmin in enumerate(xmin_vec):
        for j, xmax in enumerate(xmax_vec):
            xmin_is_smaller = xmin < xmax
            interval_has_data = np.sum((X<xmin) | (xmax<X)) > 2
            interval_is_long = xmax/xmin >= interval_length_threshold
            if xmin_is_smaller and interval_has_data and interval_is_long:
                alpha_mat[i,j] = src.estexp(X, xmin, xmax, X_dattype)
                KS_mat[i,j] = src.estKS(X, xmin, xmax, alpha_mat[i,j], X_dattype)
            else:
                alpha_mat[i,j] = np.inf
                KS_mat[i,j] = np.inf

    # Add penalty to KS metric values
    penalty = (-1)*pen_slope * \
                        np.log(xmin_vec.transpose()**(-1) * xmax_vec);
    pKS_mat = KS_mat + penalty

    # Find the index of the minimum KS metric (2D matrix) value
    i, j = np.unravel_index(pKS_mat.argmin(), (nr_xmins, nr_xmaxs))

    # Return the power-law fit: alpha, xmin, xmax, KS-, pKS-metric values
    return alpha_mat[i,j], xmin_vec[i], xmax_vec[j], KS_mat[i,j], pKS_mat[i,j]
