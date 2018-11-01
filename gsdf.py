# ISSUES
# 1 - What is the best way to turn numbers into engineering formatted string?
# 2 - Plotting approximate and theoretical PDF for a random sample should be implemented next


import numpy as np

def gsdf(*varargin):
    # GSDF generates synthetic data using CDF.

    # Input arguments
    type = varargin[0]
    n = varargin[3]
    plot_log_log_pdf = varargin[4]

    # Draw a random seed and set the random number generator
    rng_seed = np.random.randint(1,1000000)
    np.random.seed(rng_seed)

    if type == 'EPL1':
        # Exact power-law in an interval
        alpha = varargin[1]     # power-law exponent
        xmin = varargin[2][0]   # power-law upper bound
        M = varargin[2][1]      # power-law lower bound

        data_title = f"EPL1({n}pts): PL({alpha}) in [{xmin}, {M}]"

        C = (1-alpha) / ( M**(1-alpha) - xmin**(1-alpha) );

        U = np.random.rand(n)
        T = list( map( lambda x : \
            ((1-alpha)*x/C + xmin**(1-alpha))**(1/(1-alpha)) , U ) )

    else:
        print('Unexpected data name')

# def calc_PDF_vals(type, PDFx):
#     # Calculate PDF values at given points for synthetic distributions
#
#     # Initialize an array for the PDF values
#     PDFn = np.zeros(len(PDFx))
#
#     if type == 'EPL1':
#         PDFn[ PDFx<xmin ]
