import numpy as np
import matplotlib.pyplot as plt

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
            else:
                raise UnexpectedInputs
        except UnexpectedInputs:
            print('Input:', varargin[i])
            print('Error in <papod.py>: Unexpected inputs!')
            import sys
            sys.exit(1)
        i = i+2

    # Check Inputs
    try:
        if 0 in X:
            raise DataSetContainsZero
    except:
        print('Error in <papod.py>: Your data set contains zero!')
        import sys
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
    plt.loglog(Xx, Xn, basex=np.e, basey=np.e)
    plt.draw()
    plt.show()
