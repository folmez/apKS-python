import numpy as np
import src
import samples
import pytest

def test_synthetic_data_shape():
    assert samples.X_EPL1.shape == (len(samples.X_EPL1), )
    assert samples.X_EPL2.shape == (len(samples.X_EPL2), )
    assert samples.X_EPL3.shape == (len(samples.X_EPL3), )

def test_synthetic_data_generation_errors():
    plot_sample = False
    with pytest.raises(Exception):
        src.gsdf('EPL1', samples.alpha_EPL2, samples.bounds_EPL2, \
                            samples.n_EPL2, plot_sample)
