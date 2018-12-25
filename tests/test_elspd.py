import numpy as np
import src

# Generate numbers from 1 to 100
X = np.linspace(1.0, 100.0, num=100, endpoint=True)

def test_elspd_output_shape():
    # When m = 4, should return a numpy-array of shape (8,1)
    assert np.shape(src.elspd(X, 4)) == (1, 8)

    # When m =100, should return a numpy-array of shape (100,1)
    assert np.shape(src.elspd(X, 100)) == (1, 100)

def test_whether_elspd_works_correctly():
    # When m = 4, should return [1,2,4,7,14,27,52,100]
    np.testing.assert_array_equal(\
            src.elspd(X, 4), [[1.0,2.0,4.0,7.0,14.0,27.0,52.0,100.0]])

    # When m =100, should return the original sample
    np.testing.assert_array_equal(src.elspd(X, 100), [X])
