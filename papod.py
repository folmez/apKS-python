import numpy as np
import matplotlib.pyplot as plt
import sys

# define errors
class Error(Exception):
   """Base class for other exceptions"""
   pass

class UnexpectedInputs(Error):
   """Unexpected inputs!"""
   pass

class DataSetContainsZero(Error):
   """Your data set contains zero!"""
   pass

def papod(*varargin):

    # Input arguments
    X = varargin[0];

    data_title = 'Untitled data'
    nr_bins = np.amin([round(len(X)*0.1), 50])
    density_color = 'b'
    plot_stuff = True

    i = 1
    while i<len(varargin):
        try:
            if varargin[i] == 'data_title':
                data_title = varargin[i+1]
            elif varargin[i] == 'nr_bins':
                nr_bins = varargin[i+1]
            elif varargin[i] == 'power-law fit':
                [alpha, xmin, xmax] = varargin[i+1][0:3]
                if len(varargin[i+1]) == 3:
                    von = 1 # assume statistically valid
                else:
                    von = varargin[i+1][3] # statistical validity inputted
            elif varargin[i] == 'plot_stuff':
                plot_stuff = varargin[i+1]
            else:
                raise UnexpectedInputs
        except UnexpectedInputs:
            print('Input:', varargin[i])
            print('Error in <papod.py>: Unexpected inputs!')
            sys.exit(1)
        i = i+2

    # Check Inputs
    try:
        if 0 in X:
            raise DataSetContainsZero
    except:
        print('Error in <papod.py>: Your data set contains zero!')
        sys.exit(1)

    # Model
    n = len(X)
    bin_edges = np.logspace(np.log10(min(X)), np.log10(max(X)+1e-10), nr_bins);

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
        Xx =  np.add( bin_edges[0:len(bin_edges)-1], bin_edges[1:len(bin_edges)])/2

    # Normalize
    Xn = Xn/n

    # Approximate PDF of data
    approximate_PDF_label = f'Approx PDF ({n} pts)'
    approx_PDF_fig = plt.figure()
    plt.loglog(Xx, Xn, 'b.', basex=10, basey=10, label=approximate_PDF_label)
    try:
        xmin
    except NameError:
        # xmin is not defined
        temp = 0
    else:
        i1 = np.where(Xx>=xmin)[0][0]   # first idx in the power-law region
        i2 =np.where(Xx<=xmax)[0][-1]   # last idx in the power-law region

        C_hat = np.mean( Xn[i1:i2+1] * ( Xx[i1:i2+1] ** alpha ) )
        y0 = C_hat * 1 / ( xmin ** alpha )
        y1 = C_hat * 1 / ( xmax ** alpha )
        pl_vs_data_percentage = \
                        np.round( np.sum((xmin<=X) & (X<=xmax)) / len(X) *100)
        pl_fit_title = \
                f'PL(%{pl_vs_data_percentage}):[{xmin},{xmax}], alpha={alpha}'
        plt.loglog([xmin, xmax], [y0, y1], 'ro-',\
                                    label=pl_fit_title, linewidth=2.0)
    plt.title(data_title)
    plt.legend()
    if plot_stuff:
        plt.show()

    return Xn, Xx, approx_PDF_fig
