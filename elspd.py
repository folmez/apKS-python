import numpy as np

def elspd(X, m):

    nr_pts = np.ceil(np.log10(np.amax(X)/np.amin(X))) * m
    nr_pts = int(nr_pts)

    if nr_pts >= len(X):
        LmX = X
    else:
        LmX = np.zeros(nr_pts)
        # Identify logarithmicall equally spaced points between the minimum
        # and the maximum of the data
        eq_log_spaced_pts = np.logspace(np.log10(np.amax(X)), \
                                                np.log10(np.amin(X)), nr_pts)
        # Find closest data points to the set logarithmicall equally spaced
        # points without repeating the same data point
        for i in np.arange(nr_pts):
            ind_closest_nbor = np.argmin(np.absolute(X-eq_log_spaced_pts[i]))
            LmX[i] = X[ind_closest_nbor]
            # Avoid repeating the same element
            X[ind_closest_nbor] = -1;

    return np.sort(LmX)
