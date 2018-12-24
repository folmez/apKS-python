import numpy as np
import src

def test_elspd():
    # Generate numbers from 1 to 100
    X = np.linspace(1.0, 100.0, num=100, endpoint=True)

    np.testing.assert_array_equal(X, np.arange(1,101))

    # When m = 4 should return [1,2,4,7,14,27,52,100]
    np.testing.assert_array_equal(\
            src.elspd(X, 4), [1.0,2.0,4.0,7.0,14.0,27.0,52.0,100.0])

    # When m =100 should return the original sample
    np.testing.assert_array_equal(src.elspd(X, 100), X)
