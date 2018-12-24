import sys
import numpy as np

def estexp(X, xmin, xmax, X_data_type):
    # Estimate the power-law exponent on a given interval [xmin, xmax]

    # Set of candidate power-laws
    set_of_alphas = np.arange(1.01,3.51,0.01)

    # Truncate data set
    tX = X[(xmin <= X) & (X <= xmax)]

    # A term needed in the likelihood function
    if X_data_type == 'REAL':
        U = (xmax**(1-set_of_alphas) - \
                    xmin**(1-set_of_alphas)) / (1-set_of_alphas)
    elif X_data_type == 'INTS':
        x_vals = np.arange(xmin, xmax+0.1, 1)
        U = [sum(1 / x_vals**a) for a in set_of_alphas]

    # Compute the likelihood function of discretized alpha
    L_of_alphas = (-1) * set_of_alphas * sum(np.log(tX)) - \
                                len(tX) * np.log(U)
    # Find the index of the maximum likelihood
    L_max_index = np.argmax(L_of_alphas)

    # Determine the alpha from the index
    alpha = set_of_alphas[L_max_index]

    return alpha
