import numpy as np
import pytest
import src
import samples

n = 1000

@pytest.mark.slow
def test_apKS_works_ok_for_EPL1():
    assert_apKS_works('EPL1', samples.bounds_EPL1, n)

@pytest.mark.slow
def test_apKS_works_ok_for_EPL2():
    assert_apKS_works('EPL2', samples.bounds_EPL2, n)

@pytest.mark.slow
def test_apKS_works_ok_for_EPL3():
    assert_apKS_works('EPL3', samples.bounds_EPL3, n)

def assert_apKS_works(sample_rule, bounds_pl, n):
    # Generate an array of exponents
    alpha_pl_vec = np.arange(1.5, 3.00, 0.5)

    plot_sample = False
    display_p_val_stuff = False
    relative_tolerance = 0.10
    for a in alpha_pl_vec:
        # Generate a random sample
        X = src.gsdf(sample_rule, a, bounds_pl, n, plot_sample)

        # Test KS method for bounded power-law fitting
        print("\n" + 80*"-")
        print(f" TESTING apKS on an " + sample_rule + " sample with "\
                f"alpha={a:1.2f}, " + \
                f"[xmin, xmax] = [{bounds_pl[-2]:3.2f},{bounds_pl[-1]:3.2f}]")
        print(80*"-")
        a_hat, _, _, _, _, _ = src.apKS(X, display_p_val_stuff=display_p_val_stuff)
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)
