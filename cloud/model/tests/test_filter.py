import pytest
import numpy as np
from cloud.model.algorithms.filter import median_filter, moving_average_filter

def test_median_filter():
    data = np.array([1, 3, 2, 4, 100, 5])
    filtered = median_filter(data, window_size=3)
    assert np.allclose(filtered, np.array([1., 2., 3., 4., 5., 5.]))

def test_moving_average_filter():
    data = np.array([1, 3, 2, 4, 100, 5])
    filtered = moving_average_filter(data, window_size=3)
    assert np.allclose(filtered, np.array([1.66666667, 2., 3., 35.33333333, 36.33333333, 36.66666667]))
