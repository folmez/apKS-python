"""
penKS(X, X_dattype) calculates a bounder power-law fit to the given
data set X. The penalty slope is 0 by default, i.e. the default is the
KS method for bounded power law fitting.

Algorithm:
    1) Use KS-detected or lower-bound and upper-bound
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

TWO_STEP_DETECTION_MAX_RECURSION_DEPTH = 10

def penKS(X, X_dattype, xmin_vec=0, xmax_vec=0, \
            pen_slope=0, data_title='Untitled data', \
            min_nr_trial_pts_in_a_decade=10, interval_length_threshold=10):

    # Define lower-bound and upper-bound candidates
    if xmin_vec is 0 and xmax_vec is 0:
        LmX = src.elspd(X, min_nr_trial_pts_in_a_decade)
        xmin_vec, xmax_vec = LmX, LmX

    # Step 1: Find a power-law fit with no penalty, just KS metric values
    alpha_hat, xmin_hat, xmax_hat, KS_val = find_power_law_fit(\
                            X, X_dattype, xmin_vec, xmax_vec, 0,\
                            interval_length_threshold)

    # Step 2: If a non-zero penalty slope is inputted, use it to fine-tune the power-law
    #         found in step 1 via a two-step detection approach using penalized
    #         KS metric values:
    #           a) Fix xmax_hat, detect a new lower bound xmin_hat
    #           b) Fix xmin_hat, detect a new upper bound xmax_hat
    #         Above approach eventually converges to an interval. That is the
    #         interval chosen by the penalized KS metric. Notice that (a) and
    #         (b) above are one-dimensional minimizations. Penalized KS metric
    #         in 2D is not very convenient to use, its influence on the interval
    #         is confusing. It is more convenient to use it in 1D in this way.
    if pen_slope is not 0:
        count = 0
        while True:
            # a) Fix xmax_hat, detect a new lower bound xmin_hat
            alpha_hat, new_xmin_hat, _, KS_val = find_power_law_fit(\
                            X, X_dattype, \
                            LmX[LmX<=xmax_hat], np.asarray([xmax_hat]), \
                            pen_slope, interval_length_threshold)

            # b) Fix xmin_hat, detect a new upper bound xmax_hat
            alpha_hat, _, new_xmax_hat, KS_val = find_power_law_fit(\
                            X, X_dattype, \
                            np.asarray([new_xmin_hat]), LmX[new_xmin_hat<=LmX], \
                            pen_slope, interval_length_threshold)

            count = count + 1

            # Describe when the iteration ends
            if [xmin_hat, xmax_hat] == [new_xmin_hat, new_xmax_hat]:
                # Break out when the interval converged
                break
            elif count > TWO_STEP_DETECTION_MAX_RECURSION_DEPTH:
                raise RecursionError('Two-step detection did not converge!')

    return alpha_hat, xmin_hat, xmax_hat, KS_val

def find_power_law_fit(X, X_dattype, xmin_vec, xmax_vec, \
            pen_slope, interval_length_threshold):
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

    # Calculate penalized KS metric
    pKS_mat = calc_penalized_KS_metric(KS_mat, pen_slope, xmin_vec, xmax_vec)

    # Find the index of the minimum penalized KS metric (2D matrix) value
    i, j = np.unravel_index(pKS_mat.argmin(), (nr_xmins, nr_xmaxs))

    # Return the power-law fit: alpha, xmin, xmax, KS-, pKS-metric values
    return alpha_mat[i,j], xmin_vec[i], xmax_vec[j], KS_mat[i,j]

def calc_penalized_KS_metric(KS_mat, pen_slope, xmin_vec, xmax_vec):
    # Reshape candidate arrays for penalty matric computation
    reshaped_xmin_vec = xmin_vec.reshape([1, len(xmin_vec)])
    reshaped_xmax_vec = xmax_vec.reshape([1, len(xmax_vec)])

    # Compute penalty term
    penalty = (-1) * pen_slope * \
            np.log(reshaped_xmin_vec.transpose()**(-1) * reshaped_xmax_vec)

    # Add penalty to KS metric values
    return KS_mat + penalty
