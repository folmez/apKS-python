import numpy as np
import src

def test_synthetic_data_shape():
    # Generate an EPL1 random sample
    n_pl = 1000
    xmin_pl, xmax_pl = 1, 100
    alpha_pl = 2
    plot_sample = False
    X = src.gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)

    assert X.shape == (len(X), )
