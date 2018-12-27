import src

# Generate an EPL1 random sample
n_EPL1 = 2000
xmin_EPL1, xmax_EPL1 = 1, 100
bounds_EPL1 = [xmin_EPL1, xmax_EPL1]
alpha_EPL1 = 2
plot_sample = False
X_EPL1 = src.gsdf('EPL1', alpha_EPL1, [xmin_EPL1, xmax_EPL1], n_EPL1, plot_sample)

# Generate an EPL2 random sample
n_EPL2 = 2000
t0, xmin_EPL2, xmax_EPL2 = 0.5, 1, 100
bounds_EPL2 = [t0, xmin_EPL2, xmax_EPL2]
alpha_EPL2 = 1.5
plot_sample = False
X_EPL2 = src.gsdf('EPL2', alpha_EPL2, [t0, xmin_EPL2, xmax_EPL2], n_EPL2, plot_sample)
