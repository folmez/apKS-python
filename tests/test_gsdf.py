import numpy as np
import src
import samples

def test_synthetic_data_shape():
    assert samples.X_EPL1.shape == (len(samples.X_EPL1), )
    assert samples.X_EPL2.shape == (len(samples.X_EPL2), )
