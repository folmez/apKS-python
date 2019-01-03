import sys
import numpy as np
import matplotlib.pyplot as plt
import src

"""
PAPOD(X) plots the approximate pdf of the given data set X using
logarithmic binning and plots a bounded power-law fit on top if provided.
"""

def papod(X, power_law_fit=0, nr_bins=0, data_title='Untitled data', \
                                                            plot_stuff=True):

    # Approximate PDF plot parameters
    density_color = 'b'
    if nr_bins is 0:
        nr_bins = get_default_number_of_bins(len(X))

    # Power-law fit parameters
    if power_law_fit is not 0:
        a_power_law_fit_is_inputted = True
        alpha, xmin, xmax = power_law_fit[0:3]
        # Assume statistically valid if it is not specified
        fit_is_valid = True if len(power_law_fit) == 3 else power_law_fit[3]
    else:
        a_power_law_fit_is_inputted = False
    pl_fit_linestyle = '-' if fit_is_valid else '--'

    # Data sample must not contain 0
    if 0 in X:
        raise ValueError('Data set contains zero!')

    # Model
    n = len(X)
    bin_edges = np.logspace(np.log10(min(X)), np.log10(max(X)+1e-10), nr_bins)

    if  np.array_equiv(X, np.floor(X)) and n < 500:
        # Small integer data set
        Xx = np.unique(X)
        Xn = np.zeros(len(Xx))
        pos = 0
        for i in Xx:
            Xn[pos] = X.count(i)
            pos = pos+1
    else:
        Xn, bin_edges = np.histogram(X, bins=bin_edges)
        Xn = Xn / np.diff(bin_edges)
        Xx = np.add(bin_edges[0:len(bin_edges)-1], bin_edges[1:len(bin_edges)])/2

    # Normalize
    Xn = Xn/n

    # Approximate PDF of data
    approximate_PDF_label = f'Approx PDF ({n} pts)'
    approx_PDF_fig = plt.figure()
    plt.loglog(Xx, Xn, 'b.', basex=10, basey=10, label=approximate_PDF_label)

    # Plot power-law fit according to inputted power-law fit parameters
    if a_power_law_fit_is_inputted:
        i1 = np.where(Xx >= xmin)[0][0]   # first idx in the power-law region
        i2 = np.where(Xx <= xmax)[0][-1]   # last idx in the power-law region

        C_hat = np.mean(Xn[i1:i2+1] * (Xx[i1:i2+1] ** alpha))
        y0 = C_hat * 1 / (xmin ** alpha)
        y1 = C_hat * 1 / (xmax ** alpha)
        pl_vs_data_percentage = \
                        np.round(np.sum((xmin <= X) & (X <= xmax)) / len(X) *100)
        pl_fit_title = f'PL(%{pl_vs_data_percentage:2.0f}):[{xmin:.2f},{xmax:.2f}], alpha={alpha:1.2f}'
        plt.loglog([xmin, xmax], [y0, y1], 'ro'+pl_fit_linestyle,\
                                    label=pl_fit_title, linewidth=2.0)

    # Put data title, legend on plot and then show the plot
    plt.title(data_title)
    plt.legend()
    if plot_stuff:
        plt.show()

    return Xn, Xx, approx_PDF_fig

def get_default_number_of_bins(n):
    return np.amin([round(n*0.1), src.MAX_NUM_OF_BINS])
