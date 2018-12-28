"""
ESTPVAL(X, alpha, xmin, xmax, X_dattype, KS, nr_reps) estimates the
p-value corresponding to the power-law fit with power-law exponent alpha
over the interval [xmin, xmax] with the KS distance value KS using
nr_reps semiparametric bootstrap samples.
"""

import numpy as np
import time
import src

def estpval(X, X_dattype, alpha, xmin, xmax, qof_val, nr_reps=25, \
            display_p_val_stuff=True, min_nr_trial_pts_in_a_decade=10, \
            interval_length_threshold=10):

    # Make sure power-law exponent is positive
    alpha = abs(alpha)

    # Calculate an upper limit for KS metric value above which the power-law fit
    # will be automatically rejected as invalid without p-value estimation
    nr_data_points = np.sum((xmin<=X) & (X<=xmax))
    reasonable_KS_upper_limit = 20 / np.sqrt(nr_data_points)

    # Estimate p-value
    if qof_val < reasonable_KS_upper_limit:
        # Initialize a boolean array for whether the KS metric values of
        # power-law fits to semiparametric bootstrap samples are larger than
        # the inputted KS value
        sb_qof_bool = np.full(nr_reps, True)

        # Display p-value estimation header
        if display_p_val_stuff:
            print()
            print("[0]\t", "p-val\t", "Time\t", \
                    "Bounds({:3.2f},{:3.2f})\t".format(xmin, xmax), \
                    "KS({:1.4f})\t".format(qof_val), \
                    "alpha({:1.2f})".format(alpha))

        tic = time.time()
        for i in range(nr_reps):
            # Construct a semiparametric bootstrap sample from X
            sb_X = src.gsbd(X, alpha, xmin, xmax, X_dattype)

            # Determine a power-law-fit based on KS metric
            sb_pl_alpha, sb_pl_xmin, sb_pl_xmax, sb_pl_KS_val = \
                src.penKS(sb_X, X_dattype, pen_slope=0, \
                min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                interval_length_threshold=interval_length_threshold)

            # Record the KS metric value
            sb_qof_bool[i] = sb_pl_KS_val >= qof_val

            # Display most recent semiparamteric bootstrap sample info
            if display_p_val_stuff:
                print("[{}]\t".format(i+1), \
                "{:1.4f}".format(np.sum(sb_qof_bool[:i+1]) / (i+1)), \
                "[{:2.2f}m]\t".format((time.time()-tic)/60), \
                "Bounds({:3.2f},{:3.2f})\t".format(sb_pl_xmin, sb_pl_xmax), \
                "KS({:1.4f})\t".format(sb_pl_KS_val), \
                "alpha({:1.2f})".format(sb_pl_alpha))

        p_val = np.sum(sb_qof_bool) /  nr_reps

    else:
        if display_p_val_stuff:
            print("KS-value ({:3.4f}) of the ".format(qof_val) + \
                    "power-law fit ({:1.2f},{:1.2f}) ".format(xmin, xmax) + \
                    "is too high. (>{:3.4f})".format(reasonable_KS_upper_limit))
            print("Rejected without estimating p-value")
        p_val = np.inf

    return p_val
