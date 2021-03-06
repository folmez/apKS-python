import src

n = 5000

# Generate an EPL1 random sample
n_EPL1 = n
xmin_EPL1, xmax_EPL1 = 1.0, 100.0
bounds_EPL1 = [xmin_EPL1, xmax_EPL1]
alpha_EPL1 = 2.0
plot_sample = False
X_EPL1, X_EPL1_title = src.gsdf('EPL1', alpha_EPL1, \
                                [xmin_EPL1, xmax_EPL1], n_EPL1, plot_sample)

# Generate an EPL2 random sample
n_EPL2 = n
t0, xmin_EPL2, xmax_EPL2 = 0.5, 1.0, 100.0
bounds_EPL2 = [t0, xmin_EPL2, xmax_EPL2]
alpha_EPL2 = 1.5
plot_sample = False
X_EPL2, X_EPL2_title = src.gsdf('EPL2', alpha_EPL2, \
                                [t0, xmin_EPL2, xmax_EPL2], n_EPL2, plot_sample)

# Generate an EPL3 random sample
n_EPL3 = n
mu_EPL3, xmin_EPL3, xmax_EPL3 = -1.0, 1.0, 100.0
bounds_EPL3 = [mu_EPL3, xmin_EPL3, xmax_EPL3]
alpha_EPL3 = 1.5
plot_sample = False
X_EPL3, X_EPL3_title = src.gsdf('EPL3', alpha_EPL3, \
                                [mu_EPL3, xmin_EPL3, xmax_EPL3], n_EPL3, plot_sample)
