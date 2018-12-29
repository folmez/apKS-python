"""
Testing how python handles appending values to a list outside a function
"""

def test_python_list_appending():
    a_hats, x1_hats, x2_hats = [],[],[]

    append_power_law_fit_parameters(a_hats, x1_hats, x2_hats, 1.5,  1.,   100.)
    append_power_law_fit_parameters(a_hats, x1_hats, x2_hats, 2.0, 10.,   100.)
    append_power_law_fit_parameters(a_hats, x1_hats, x2_hats, 3.0,  0.1, 1000.)

    assert  a_hats == [1.5, 2.0, 3.0]
    assert x1_hats == [1., 10., 0.1]
    assert x2_hats == [100., 100., 1000.]

def append_power_law_fit_parameters(a_hats, x1_hats, x2_hats, a,x1,x2):
    a_hats.append(a)
    x1_hats.append(x1)
    x2_hats.append(x2)
