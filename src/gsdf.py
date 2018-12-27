# ISSUES
# 1 - What is the best way to turn numbers into engineering formatted string?

import datetime
import numpy as np
import matplotlib.pyplot as plt

import src

def gsdf(*varargin):
    # GSDF generates synthetic data using CDF.

    # Input arguments
    sample_rule = varargin[0]
    n_data = varargin[3]
    plot_log_log_pdf = varargin[4]

    # Draw a random seed and set the random number generator.
    time_now = datetime.datetime.now()
    rng_seed = time_now.hour*10000000 + time_now.microsecond
    np.random.seed(rng_seed)

    # Model
    if sample_rule == 'EPL1':
        # Exact power-law in an interval
        alpha_pl = varargin[1]     # power-law exponent
        xmin_pl = varargin[2][0]   # power-law upper bound
        M_pl = varargin[2][1]      # power-law lower bound

        # Check bounds
        if len(varargin[2]) is not 2:
            raise IncorrectBounds

        # Set random sample title
        data_title = \
                f"EPL1({n_data}pts): PL({alpha_pl}) in [{xmin_pl}, {M_pl}]"

        # Generate random sample
        C_pl = (1-alpha_pl) / (M_pl**(1-alpha_pl) - xmin_pl**(1-alpha_pl))
        U = np.random.rand(n_data)
        T = ((1-alpha_pl) * U/C_pl + xmin_pl**(1-alpha_pl)) ** (1/(1-alpha_pl))

    elif sample_rule == 'EPL2':
        # Exact power-law in an interval with sharp transitions to non-PL
        alpha_pl = varargin[1]          # power-law exponent
        t0 = 0 if len(varargin[2]) == 2 else varargin[2][0] # absolute lower-bound
        xmin_pl = varargin[2][-2]       # power-law lower bound
        M_pl = varargin[2][-1]          # power-law upper bound

        # The exponential decay in the tail
        beta = alpha_pl * np.log(M_pl/xmin_pl) / (M_pl-xmin_pl)

        # Set random sample title
        data_title = "EPL2({}pts): ".format(n_data) + \
                        "PL({}) ".format(alpha_pl) + \
                        "in [{}, {}], ".format(xmin_pl, M_pl) + \
                        "exp({}) otherwise".format(beta)

        # Compute coefficients A and C that guarantee continuity and probability
        mat_coeffs = [ [np.exp((-1)*beta*M_pl), (-1)*M_pl**((-1)*alpha_pl)],
                [(np.exp((-1)*beta*t0)+np.exp((-1)*beta*M_pl)-np.exp((-1)*beta*xmin_pl)) / beta, \
                 (M_pl**(1-alpha_pl)-xmin_pl**(1-alpha_pl)) / (1-alpha_pl)] ]

        # 0 - continuity condition, 1 - probability cond
        [A, C] = np.linalg.solve(mat_coeffs, [0, 1])

        # Generate random sample by inverse sampling
        U = np.random.rand(n_data)
        T = np.zeros(n_data)
        CDF_at_xmin = A/(-beta)*(np.exp(-beta*xmin_pl)-np.exp(-beta*t0))
        CDF_at_M = CDF_at_xmin + C/(1-alpha_pl) * \
                                    (M_pl**(1-alpha_pl)-xmin_pl**(1-alpha_pl))
        idx = U < CDF_at_xmin
        T[idx] = (-1)/beta * np.log((-1)*beta/A*U[idx] + np.exp((-1)*beta*t0))
        idx = (U <= CDF_at_M) & (U >= CDF_at_xmin)
        T[idx] = ((1-alpha_pl)/C*(U[idx]-CDF_at_xmin) + \
                                    xmin_pl**(1-alpha_pl))**(1/(1-alpha_pl))
        idx = U > CDF_at_M
        T[idx] = (-1)/beta * np.log(-beta/A*(U[idx]-CDF_at_M) + \
                                                            np.exp(-beta*M_pl))

    elif sample_rule == 'EPL3':
        # Exact power-law in an interval with smooth transitions to non-PL
        alpha_pl = varargin[1]          # power-law exponent
        mu, xmin_pl, M_pl = varargin[2] # log-normal(mu,sigma) and power-law bounds

        # continuous slope at xmin_pl
        try:
            sigma = np.sqrt((np.log(xmin_pl)-mu)/(alpha_pl-1))
        except:
            raise EPL3_ERROR
        # continuous slope at M_pl
        beta = alpha_pl/M_pl

        # Set random sample title
        data_title = "EPL3({}pts): ".format(n_data) + \
                        "0 < log-n({},{}) < ".format(mu, sigma) + \
                        "PL({}) < ".format(alpha_pl) + \
                        "{} < ".format(M_pl) + \
                        "exp({})".format(beta)

        # Compute coefficients A, C and L for continuity and probability
        mat_coeffs = [[np.exp((-1)*beta*M_pl) , (-1)*M_pl**((-1)*alpha_pl) , 0], \
                        [0 , xmin_pl**((-1)*alpha_pl) , (-1)*src.my_lognorm_pdf(xmin_pl,mu,sigma)], \
                            [np.exp((-1)*beta*M_pl)/beta , \
                            (M_pl**(1-alpha_pl)-xmin_pl**(1-alpha_pl))/(1-alpha_pl) , \
                            src.my_lognorm_cdf(xmin_pl,mu,sigma)]]
        [A, C, L] = np.linalg.solve(mat_coeffs, [0, 0, 1])

        # Generate random sample by inverse sampling
        U = np.random.rand(n_data)
        T = np.zeros(n_data)
        CDF_at_xmin = L * src.my_lognorm_cdf(xmin_pl, mu, sigma)
        CDF_at_M = CDF_at_xmin + \
                    C * (M_pl**(1-alpha_pl)-xmin_pl**(1-alpha_pl))/(1-alpha_pl)
        T[U<=CDF_at_xmin] = src.my_lognorm_inv_cdf( \
                                            U[U<=CDF_at_xmin]*1/L, mu, sigma)
        T[(U<=CDF_at_M) & (U>CDF_at_xmin)] = \
                    ( (U[(U<=CDF_at_M) & (U>CDF_at_xmin)] - CDF_at_xmin) * \
                        (1-alpha_pl)/C + xmin_pl**(1-alpha_pl) )**(1/(1-alpha_pl))
        T[U>CDF_at_M] = (-1)/beta * np.log( (-1) * (beta/A) * \
                        (U[U>CDF_at_M] - CDF_at_M - A*np.exp(-beta*M_pl)/beta) )

    else:
        raise UNEXPECTED_SAMPLE_NAME_ERROR

    # Plot
    if plot_log_log_pdf:
        # Calculate and plot empirical PDF
        _, PDFx, ePDF_fig = src.papod(T, 'data_title', data_title, \
                                                        'plot_stuff', False)

        # Calculate true PDF
        tPDFn = np.zeros(len(PDFx))
        if sample_rule == 'EPL1':
            tPDFn[PDFx < xmin_pl] = 0
            tPDFn[(xmin_pl <= PDFx) & (PDFx < M_pl)] = \
            C_pl / (PDFx[(xmin_pl <= PDFx) & (PDFx <= M_pl)] ** alpha_pl)
            tPDFn[M_pl <= PDFx] = 0
        elif sample_rule == 'EPL2':
            tPDFn[PDFx < xmin_pl] = A*np.exp((-1)*beta*PDFx[PDFx<xmin_pl])
            tPDFn[(xmin_pl <= PDFx) & (PDFx < M_pl)] = C * \
                (PDFx[(xmin_pl <= PDFx) & (PDFx < M_pl)])**((-1)*alpha_pl)
            tPDFn[M_pl <= PDFx] = A*np.exp((-1)*beta*PDFx[M_pl <= PDFx])
        elif sample_rule == 'EPL3':
            tPDFn[PDFx < xmin_pl] =  L * \
                        src.my_lognorm_pdf(PDFx[PDFx < xmin_pl], mu, sigma)
            tPDFn[(xmin_pl <= PDFx) & (PDFx < M_pl)] = C * \
                (PDFx[(xmin_pl <= PDFx) & (PDFx < M_pl)])**((-1)*alpha_pl)
            tPDFn[M_pl <= PDFx] = A * np.exp((-1)*beta*PDFx[M_pl <= PDFx])


        # Plot true PDF
        theoretical_PDF_title = 'Theoretical PDF'
        plt.figure(ePDF_fig.number)
        plt.loglog(PDFx, tPDFn, 'b', label=theoretical_PDF_title, linewidth=2.0)
        plt.legend()
        plt.draw()
        plt.show()

    # Return a sorted random sample
    return np.sort(T)

### ERRORS

class Error(Exception):
    """Base class for other exceptions"""
    pass

class IncorrectBounds(Error):
    """Raised when inputted bounds for random sample generation are off"""
    pass

class UNEXPECTED_SAMPLE_NAME_ERROR(Error):
    """Raised when inputted sample name is wrong"""

class EPL3_ERROR(Error):
    """Raised when something goes wrong in the generation of an EPL3 sample"""
