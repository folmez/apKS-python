# ISSUES
# 1 - What is the best way to turn numbers into engineering formatted string?

import numpy as np
import matplotlib.pyplot as plt
import datetime
from papod import papod

def gsdf(*varargin):
    # GSDF generates synthetic data using CDF.

    # Input arguments
    data_type = varargin[0]
    n = varargin[3]
    plot_log_log_pdf = varargin[4]

    # Draw a random seed and set the random number generator.
    time_now = datetime.datetime.now()
    rng_seed = time_now.hour*10000000 + time_now.microsecond
    np.random.seed(rng_seed)

    # Model
    if data_type == 'EPL1':
        # Exact power-law in an interval
        alpha = varargin[1]     # power-law exponent
        xmin = varargin[2][0]   # power-law upper bound
        M = varargin[2][1]      # power-law lower bound
        # Set random sample title
        data_title = f"EPL1({n}pts): PL({alpha}) in [{xmin}, {M}]"

        # Generate random sample
        C = (1-alpha) / ( M**(1-alpha) - xmin**(1-alpha) );
        U = np.random.rand(n)
        T = ( (1-alpha) * U/C + xmin**(1-alpha) ) ** ( 1/(1-alpha) )

    else:
        print('Unexpected data name')

    # Plot
    if plot_log_log_pdf:
        # Calculate and plot empirical PDF
        ePDFn, PDFx, ePDF_fig = papod(T, 'data_title', data_title, \
                                                        'plot_stuff', False)

        # Calculate true PDF
        tPDFn = np.zeros(len(PDFx))
        if data_type == 'EPL1':
            tPDFn[ PDFx<xmin ] = 0;
            tPDFn[ (xmin<=PDFx) & (PDFx<M) ] = \
            C / ( PDFx[ (xmin<=PDFx) & (PDFx<=M) ] ** alpha )
            tPDFn[ M<=PDFx ] = 0

        # Plot true PDF
        theoretical_PDF_title = 'Theoretical PDF'
        plt.figure(ePDF_fig.number)
        plt.loglog(PDFx, tPDFn, 'b', label=theoretical_PDF_title, linewidth=2.0)
        plt.legend()
        plt.draw()
        plt.show()

    # Return random sample
    return T
