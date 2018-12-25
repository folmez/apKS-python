import numpy as np

# ESTKS(X, xmin, xmax, alpha, X_dattype) computes the KS distance
# between a power-law fit and data X
def estKS(X, xmin, xmax, alpha, X_dattype):
    min_nr_data_pts_allowed = 10
    # Truncate sample
    tX = X[(xmin<=X) & (X<=xmax)]
    # Make exponent a float
    alpha = float(alpha)

    if len(tX) < min_nr_data_pts_allowed:
        ks_stat = np.inf;
    else:
        # Left and Right empirical CDFs
        tX_left_CDF = np.arange(len(tX)) / len(tX)
        tX_right_CDF = np.arange(1,len(tX)+1) / len(tX)
        # Compute theoretical PDF
        abs_alpha = np.absolute(alpha)
        if X_dattype == 'REAL':
            if abs_alpha == 1:
                theoretical_pl_CDF = \
                    (np.log(tX)-np.log(xmin)) / (np.log(xmax)-np.log(xmin))
            else:
                theoretical_pl_CDF = \
                    1 / (xmax**(1-abs_alpha) - xmin**(1-abs_alpha)) * \
                    (tX**(1-abs_alpha) - xmin**(1-abs_alpha))
        elif X_dattype == 'INTS':
            S = np.sum(1 / tX**abs_alpha)
            theoretical_pl_CDF = np.cumsum(1 / tX**abs_alpha) / S
        # Compute KS distance
        ks_stat = np.amax(\
            [np.amax(np.absolute(tX_left_CDF - theoretical_pl_CDF)), \
            np.amax(np.absolute(tX_right_CDF - theoretical_pl_CDF))])

    return ks_stat
