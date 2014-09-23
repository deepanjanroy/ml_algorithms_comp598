import unittest
import numpy as np
from numpy import testing
from scales import Scale


class TestScales(unittest.TestCase):
    def test_unit_scale(self):
        scale = Scale.unit_scale(3)
        a = np.array([1,2,3])
        scaled_a = scale.scale(a)
        testing.assert_array_equal(a, scaled_a)

    def test_mul(self):
        scale = Scale.unit_scale(3)
        two_scale = scale * 2
        a = np.array([1,2,3])
        scaled_a = two_scale.scale(a)
        testing.assert_array_equal(a*2, scaled_a)

    def test_basic(self):
        factors = [2,3,4];
        scale = Scale(np.array(factors))
        ones = np.ones(3)
        scaled_a = scale.scale(ones)
        testing.assert_array_equal(factors, scaled_a)

    def test_less_that_one(self):
        factors = [0.1,0.2,0.3]
        scale = Scale(np.array(factors))
        ones = np.ones(3)
        scaled_a = scale.scale(ones)
        testing.assert_array_equal(factors, scaled_a)

    def test_scale_from_data(self):
        data = [[1,2,3],
                [2,-1,-4]]

        expected_scale = np.array([1.0/2.0, 1.0/2.0, 1.0/4.0])

        calculated_scale = Scale.scale_from_data(data)
        testing.assert_array_equal(expected_scale, calculated_scale.factors)

    def test_scale_on_2D_arrays(self):
        a = np.array([[1,2,3], [2,3,4]])
        scale = Scale(np.array([0.5, 2, 0]))
        scaled_a = scale(a)
        testing.assert_array_equal(scale(a[0]), scaled_a[0])
        testing.assert_array_equal(scale(a[1]), scaled_a[1])



if __name__ == "__main__":
    unittest.main()