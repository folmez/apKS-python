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
            need_p_val = 1):

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
    alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes = \
    fit_validate_record(0, alpha_hats, xmin_hats, xmax_hats,
                        qof_values, p_vals, slopes, \
                        X, X_dattype, 0, data_title, \
                        min_nr_trial_pts_in_a_decade, display_p_val_stuff, \
                        interval_length_threshold, nr_reps)

    # If the power-law fit for no penalty is not valid, terminate here
    if p_vals[-1]  <= src.P_VAL_THRESHOLD:
        plot_fit(plot_best_pl_fit, X, data_title, alpha_hats[-1], \
                        xmin_hats[-1], xmax_hats[-1], p_vals[-1])
        return alpha_hats[-1], xmin_hats[-1], xmax_hats[-1], qof_values[-1], p_vals[-1], slopes[-1]

    # Continue if penalized approach is needed
    if not need_only_KS:
        # Initiation: MINIMUM
        alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes = \
        fit_validate_record(1, alpha_hats, xmin_hats, xmax_hats, \
                            qof_values, p_vals, slopes, \
                            X, X_dattype, min_slope, data_title, \
                            min_nr_trial_pts_in_a_decade, display_p_val_stuff, \
                            interval_length_threshold, nr_reps)
        # Power-law fit corresponding to minimum penalty coefficient must be
        # identical to the one found for KS method
        if not these_fits_are_identical(\
                                [alpha_hats[0], xmin_hats[0], xmax_hats[0]], \
                                [alpha_hats[1], xmin_hats[1], xmax_hats[1]]):
            raise ValueError('Minimum penalty slope did not replicate ' + \
                                        'KS method estimated power-law fit')

        # Initiation: MAXIMUM
        while True:
            alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes = \
            fit_validate_record(2, alpha_hats, xmin_hats, xmax_hats, \
                                qof_values, p_vals, slopes, \
                                X, X_dattype, max_slope, data_title, \
                                min_nr_trial_pts_in_a_decade, display_p_val_stuff, \
                                interval_length_threshold, nr_reps)
            # Power-law fit corresponding to maximum penalty coefficient must be
            # an invalid power-law fit.
            if p_vals[-1]  <= src.P_VAL_THRESHOLD:  # invalid power-law fit
                # This is the expecte case, continue as normal
                break
            elif p_vals[-1]  > src.P_VAL_THRESHOLD: # valid power-law fit
                #   (i)  either the sample can entirely be fitted with a
                #        bounded power-law
                #   (ii) or a larger maximum slope should be inputted
                if xmin_hats[-1] <= np.amin(X) and np.amax(X) <= xmax_hats[-1]:
                    # The sample can entirely be fitted with a bounded power-law
                    plot_fit(plot_best_pl_fit, X, data_title, alpha_hats[-1], \
                                    xmin_hats[-1], xmax_hats[-1], p_vals[-1])
                    print_summary(alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes, -1)
                    return alpha_hats[-1], xmin_hats[-1], xmax_hats[-1], qof_values[-1], p_vals[-1], slopes[-1]
                else:
                    # need a larger maximum slope
                    max_slope = max_slope * 2
                    # remove the power-law fit
                    alpha_hats.pop()
                    xmin_hats.pop()
                    xmax_hats.pop()
                    qof_values.pop()
                    p_vals.pop()
                    slopes.pop()
                    print('Maximum penalty slope did not produce ' + \
                                        ' an invalid power-law fit. ' + \
                                        'Consider manually inputting ' + \
                                        'a larger maximum penalty coefficient.')

        # Loop
        slope_count = 3
        while (max_slope-min_slope) / max_slope >= slope_diff_tol:
            # Calculate the penalty slope to be tested next
            current_slope = np.sqrt(min_slope*max_slope)
            alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes = \
            fit_validate_record(slope_count, alpha_hats, xmin_hats, xmax_hats, \
                                qof_values, p_vals, slopes, \
                                X, X_dattype, current_slope, data_title, \
                                min_nr_trial_pts_in_a_decade, display_p_val_stuff, \
                                interval_length_threshold, nr_reps)

            # Increment slope count by one for every fit
            slope_count = slope_count + 1

            # Update the minimum or the maximum slope accordingly
            if p_vals[-1] <= src.P_VAL_THRESHOLD:    # invalid power-law fit
                max_slope = current_slope
            elif p_vals[-1] > src.P_VAL_THRESHOLD:   # valid power-law fit
                min_slope = current_slope

        # The last minimum slope (aka last good slope that produced a valid fit)
        # will give us the apKS method obtained valid power-law fit
        idx = slopes.index(min_slope)

        # Plot power-law fit on approximate PDF, print summary and terminate
        plot_fit(plot_best_pl_fit, X, data_title, alpha_hats[idx], \
                                    xmin_hats[idx], xmax_hats[idx], p_vals[idx])
        print_summary(alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes, idx)
        return alpha_hats[idx], xmin_hats[idx], xmax_hats[idx], qof_values[idx], p_vals[idx], slopes[idx]

def fit_validate_record(slope_count, alpha_hats, xmin_hats, xmax_hats, \
                        qof_values, p_vals, slopes, \
                        X, X_dattype, penalty_slope, data_title, \
                        min_nr_trial_pts_in_a_decade, display_p_val_stuff, \
                        interval_length_threshold, nr_reps):

        # Print slope information and increment slope count
        print(f"\nSlope #{slope_count} = {penalty_slope:.6f}")

        # Apply penalized KS method with the given penalty slope
        alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, X_dattype, \
                pen_slope=penalty_slope, data_title=data_title, \
                min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                interval_length_threshold=interval_length_threshold)

        # Check the validity of the power-law fit
        if this_power_law_fit_is_new(alpha_hat, xmin_hat, xmax_hat, \
                                        alpha_hats, xmin_hats, xmax_hats):
            print(f"A new power-law fit in [{xmin_hat:.2f}, {xmax_hat:.2f}]")
            p_val = src.estpval(X, X_dattype, \
                        alpha_hat, xmin_hat, xmax_hat, KS_val, \
                        nr_reps=nr_reps, \
                        display_p_val_stuff=display_p_val_stuff, \
                        min_nr_trial_pts_in_a_decade=min_nr_trial_pts_in_a_decade, \
                        interval_length_threshold=interval_length_threshold)
        else:
            print(f"Not a new power-law fit in [{xmin_hat:.2f}, {xmax_hat:.2f}]")
            p_val = get_p_val_from_previous_estimations( \
                                    alpha_hat, xmin_hat, xmax_hat, \
                                    alpha_hats, xmin_hats, xmax_hats, p_vals)

        # Print validity of the power-law
        print(f"Power-law exponent is {alpha_hat:.2f}")
        print_validity(p_val)

        # Record output
        alpha_hats.append(alpha_hat)
        xmin_hats.append(xmin_hat)
        xmax_hats.append(xmax_hat)
        qof_values.append(KS_val)
        p_vals.append(p_val)
        slopes.append(penalty_slope)

        # Return output
        return alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes

def this_power_law_fit_is_new(a, x1, x2, alpha_hats, xmin_hats, xmax_hats):
    # Check if this power-law has appeared before
    for a_h, x1_h, x2_h in zip(alpha_hats, xmin_hats, xmax_hats):
        if these_fits_are_identical([a_h, x1_h, x2_h], [a, x1, x2]):
            return False

    # If the loop end without a hit, return False
    return True

def get_p_val_from_previous_estimations(a, x1, x2, \
                                    alpha_hats, xmin_hats, xmax_hats, p_vals):
    for a_h, x1_h, x2_h, p_val in zip(alpha_hats, xmin_hats, xmax_hats, p_vals):
        if these_fits_are_identical([a_h, x1_h, x2_h], [a, x1, x2]):
            return p_val

def these_fits_are_identical(a, b):
    return np.allclose(a, b, atol=1e-10, rtol=0)

def plot_fit(plot_best_pl_fit, X, data_title, a, x1, x2, p_val):
    if plot_best_pl_fit:
        src.papod(X, data_title=data_title, \
                    power_law_fit=[a, x1, x2, p_val > src.P_VAL_THRESHOLD])

def print_summary(alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals, slopes, i):
    print("\n\tKS_slope\talpha\tx_min\tx_max\tqof_val\tp_val")
    for s, a, x1, x2, KS, p in zip(slopes, alpha_hats, xmin_hats, xmax_hats, qof_values, p_vals):
        print(f"\t{s:1.5f}\t\t{a:1.2f}\t{x1:3.2f}\t{x2:1.1e}\t{KS:3.4f}\t{p:1.2f}")

    print("\nThe power-law found by the apKS method has " + \
                f"exponent {alpha_hats[i]:1.2f} " + \
                f"in interval[{xmin_hats[i]:3.2f}, {xmax_hats[i]:3.2f}]")

def print_validity(p_val):
    if p_val <= src.P_VAL_THRESHOLD:    # invalid power-law fit
        print("Rejected power-law fit")
    elif p_val > src.P_VAL_THRESHOLD:   # valid power-law fit
        print("Accepted power-law fit")
