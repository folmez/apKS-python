import src
import samples
import pytest

def test_number_of_bins():
    assert src.get_default_number_of_bins(1000) == src.MAX_NUM_OF_BINS
    assert src.get_default_number_of_bins(src.MAX_NUM_OF_BINS * 10) == \
                                                            src.MAX_NUM_OF_BINS
    assert src.get_default_number_of_bins(100) == 10

def test_papod_errors():
    append_zero_and_test_error(samples.X_EPL1)
    append_zero_and_test_error(samples.X_EPL2)
    append_zero_and_test_error(samples.X_EPL3)
def append_zero_and_test_error(X):
    with pytest.raises(Exception):
        src.papod(X.append(0.0))

@pytest.mark.slow
def test_papod():
    # Plot Test 1 - Valid power-law fit on top of approximate PDF
    test1_title = '[PLOT TEST]\n'
    plot_sample(samples.X_EPL1, test1_title + samples.X_EPL1_title, \
            [samples.alpha_EPL1, samples.xmin_EPL1, samples.xmax_EPL1], \
            src.get_default_number_of_bins(len(samples.X_EPL1)))
    plot_sample(samples.X_EPL2, test1_title + samples.X_EPL2_title, \
            [samples.alpha_EPL2, samples.xmin_EPL2, samples.xmax_EPL2], \
            src.get_default_number_of_bins(len(samples.X_EPL2)))
    plot_sample(samples.X_EPL3, test1_title + samples.X_EPL1_title, \
            [samples.alpha_EPL3, samples.xmin_EPL3, samples.xmax_EPL3], \
            src.get_default_number_of_bins(len(samples.X_EPL3)))

    # Plot Test 2 - Invalid power-law fit on top of approximate PDF
    test2_title = '[PLOT TEST with INVALID POWER-LAW FIT] \n'
    plot_sample(samples.X_EPL1, test2_title + samples.X_EPL1_title, \
            [samples.alpha_EPL1 - 0.5, samples.xmin_EPL1, samples.xmax_EPL1, False], \
            src.get_default_number_of_bins(len(samples.X_EPL1)))
    plot_sample(samples.X_EPL2, test2_title + samples.X_EPL2_title, \
            [samples.alpha_EPL2 - 0.5, samples.xmin_EPL2, samples.xmax_EPL2, False], \
            src.get_default_number_of_bins(len(samples.X_EPL2)))
    plot_sample(samples.X_EPL3, test2_title + samples.X_EPL1_title, \
            [samples.alpha_EPL3 - 0.5, samples.xmin_EPL3, samples.xmax_EPL3, False], \
            src.get_default_number_of_bins(len(samples.X_EPL3)))

    # Plot Test 3 - Invalid power-law fit on top of approximate PDF
    test3_title = '[PLOT TEST with MORE BINS] \n'
    plot_sample(samples.X_EPL1, test3_title + samples.X_EPL1_title, \
            [samples.alpha_EPL1, samples.xmin_EPL1, samples.xmax_EPL1], \
            2 * src.MAX_NUM_OF_BINS)
    plot_sample(samples.X_EPL2, test3_title + samples.X_EPL2_title, \
            [samples.alpha_EPL2, samples.xmin_EPL2, samples.xmax_EPL2], \
            2 * src.MAX_NUM_OF_BINS)
    plot_sample(samples.X_EPL3, test3_title + samples.X_EPL1_title, \
            [samples.alpha_EPL3, samples.xmin_EPL3, samples.xmax_EPL3], \
            2 * src.MAX_NUM_OF_BINS)
def plot_sample(X, data_title, power_law_fit, nr_bins):
    _, _, _ = src.papod(X, data_title=data_title, power_law_fit=power_law_fit, \
                            nr_bins=nr_bins)
