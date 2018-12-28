"""
APKS(X) calculates the bounded power-law fit to the given data set X
using the apKS method described in (CITE EPL)
"""

import numpy as np
import src

def apKS(X, data_title='Untitled data', plot_best_pl_fit=1, display_stuff=1, \
            display_p_val_stuff=1, nr_reps=25, slope_diff_tol=1e-2, \
            min_nr_trial_pts_in_a_decade=10, need_only_KS=0, \
            assume_data_is_real=0, interval_length_threshold=10, \
            need_p_val = 1, p_val_threshold = 0.10):

    alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes = [],[],[],[],[],[]

    # What is the data type? Continous data ('REAL') or discrete data ('INTS')
    X_dattype = 'REAL'

    # Define a minimum penalty slope small enough for which the power-law fit found
    # by the penalized KS method is the same as the one found with the KS method.
    min_slope = 1e-5

    # Define a maximum penalty slope large enough for which the power-law fit
    # found by the penalized KS method is invalidated by the p-value test
    max_slope = 1

    # Sanity check: Apply the KS method, i.e. penalized KS method with 0 penalty
    alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, X_dattype, \
                pen_slope=0, data_title=data_title, \
                min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                interval_length_threshold=interval_length_threshold)

    # Sanity check - continued: Check the validity of the power-law fit
    p_val = src.estpval(X, X_dattype, \
                        alpha_hat, xmin_hat, xmax_hat, KS_val, \
                        nr_reps=nr_reps, \
                        display_p_val_stuff=display_p_val_stuff, \
                        min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                        interval_length_threshold=interval_length_threshold)

    # Record sanity check
    alpha_hats.append(alpha_hat)
    xmin_hats.append(xmin_hat)
    xmax_hats.append(xmax_hat)
    qof_values.append(KS_val)
    p_vals.append(p_val)
    slopes.append(0)

    # Continue if penalized approach is needed
    if not need_only_KS:
        # Initiation: MINIMUM

        # Apply penalized KS method with the minimum penalty
        alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, X_dattype, \
                    pen_slope=min_slope, data_title=data_title, \
                    min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                    interval_length_threshold=interval_length_threshold)

        # Check the validity of the power-law fit
        p_val = src.estpval(X, X_dattype, \
                    alpha_hat, xmin_hat, xmax_hat, KS_val, \
                    nr_reps=nr_reps, \
                    display_p_val_stuff=display_p_val_stuff, \
                    min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                    interval_length_threshold=interval_length_threshold)

        # Record sanity check
        alpha_hats.append(alpha_hat)
        xmin_hats.append(xmin_hat)
        xmax_hats.append(xmax_hat)
        qof_values.append(KS_val)
        p_vals.append(p_val)
        slopes.append(min_slope)

        # Initiation: MAXIMUM

        # Apply penalized KS method with the maximum penalty
        alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, X_dattype, \
                    pen_slope=max_slope, data_title=data_title, \
                    min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                    interval_length_threshold=interval_length_threshold)

        # Check the validity of the power-law fit
        p_val = src.estpval(X, X_dattype, \
                    alpha_hat, xmin_hat, xmax_hat, KS_val, \
                    nr_reps=nr_reps, \
                    display_p_val_stuff=display_p_val_stuff, \
                    min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                    interval_length_threshold=interval_length_threshold)

        # Record sanity check
        alpha_hats.append(alpha_hat)
        xmin_hats.append(xmin_hat)
        xmax_hats.append(xmax_hat)
        qof_values.append(KS_val)
        p_vals.append(p_val)
        slopes.append(max_slope)

        # Loop
        while (max_slope-min_slope) / max_slope >= slope_diff_tol:
            # Calculate the penalty slope to be tested next
            current_slope = np.sqrt(min_slope*max_slope)

            # Apply penalized KS method with the current penalty slope
            alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, X_dattype, \
                    pen_slope=current_slope, data_title=data_title, \
                    min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                    interval_length_threshold=interval_length_threshold)

            # Check the validity of the power-law fit
            p_val = src.estpval(X, X_dattype, \
                        alpha_hat, xmin_hat, xmax_hat, KS_val, \
                        nr_reps=nr_reps, \
                        display_p_val_stuff=display_p_val_stuff, \
                        min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                        interval_length_threshold=interval_length_threshold)

            # Record sanity check
            alpha_hats.append(alpha_hat)
            xmin_hats.append(xmin_hat)
            xmax_hats.append(xmax_hat)
            qof_values.append(KS_val)
            p_vals.append(p_val)
            slopes.append(current_slope)

            # Update the minimum or the maximum slope accordingly
            if p_val <= src.P_VAL_THRESHOLD:    # invalid power-law fit
                max_slope = current_slope
            elif p_val > src.P_VAL_THRESHOLD:   # valid power-law fit
                min_slope = current_slope
