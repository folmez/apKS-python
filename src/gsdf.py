# ISSUES
# 1 - What is the best way to turn numbers into engineering formatted string?

import datetime
import numpy as np
import matplotlib.pyplot as plt
import src
# from papod import papod

def gsdf(*varargin):
    # GSDF generates synthetic data using CDF.

    # Input arguments
    data_type = varargin[0]
    n_data = varargin[3]
    plot_log_log_pdf = varargin[4]

    # Draw a random seed and set the random number generator.
    time_now = datetime.datetime.now()
    rng_seed = time_now.hour*10000000 + time_now.microsecond
    np.random.seed(rng_seed)

    # Mlodel
    if data_type == 'EPL1':
        # Exact power-law in an interval
        alpha_pl = varargin[1]     # power-law exponent
        xmin_pl = varargin[2][0]   # power-law upper bound
        M_pl = varargin[2][1]      # power-law lower bound
        # Set random sample title
        data_title = \
                f"EPL1({n_data}pts): PL({alpha_pl}) in [{xmin_pl}, {M_pl}]"

        # Generate random sample
        C_pl = (1-alpha_pl) / (M_pl**(1-alpha_pl) - xmin_pl**(1-alpha_pl))
        U = np.random.rand(n_data)
        T = ((1-alpha_pl) * U/C_pl + xmin_pl**(1-alpha_pl)) ** (1/(1-alpha_pl))

    elif data_type == 'EPL2':
        # Exact power-law in an interval with sharp transitions to non-PL
        alpha_pl = varargin[1]          # power-law exponent
        # ??? Write this in one line
        if len(varargin[2]) == 2:
            t0 = 0
        elif len(varargin[2]) ==3:
            t0 = varargin[2][0]
        xmin_pl = varargin[2][-2]       # power-law lower bound
        M_pl = varargin[2][-1]          # power-law upper bound

        # The exponential decay in the tail
        beta = alpha * np.log_natural(M_pl/xmin_pl) / (M_pl-xmin_pl)

        # Set random sample title
        data_title = \
            f"EPL2({n_data}pts): PL({alpha_pl}) in [{xmin_pl}, {M_pl}], exp({beta}) otherwise"

    else:
        print('Unexpected data name')

    # Plot
    if plot_log_log_pdf:
        # Calculate and plot empirical PDF
        _, PDFx, ePDF_fig = src.papod(T, 'data_title', data_title, \
                                                        'plot_stuff', False)

        # Calculate true PDF
        tPDFn = np.zeros(len(PDFx))
        if data_type == 'EPL1':
            tPDFn[PDFx < xmin_pl] = 0
            tPDFn[(xmin_pl <= PDFx) & (PDFx < M_pl)] = \
            C_pl / (PDFx[(xmin_pl <= PDFx) & (PDFx <= M_pl)] ** alpha_pl)
            tPDFn[M_pl <= PDFx] = 0

        # Plot true PDF
        theoretical_PDF_title = 'Theoretical PDF'
        plt.figure(ePDF_fig.number)
        plt.loglog(PDFx, tPDFn, 'b', label=theoretical_PDF_title, linewidth=2.0)
        plt.legend()
        plt.draw()
        plt.show()

    # Return a sorted random sample
    return np.sort(T)
