import numpy as np

def median_filter(data, window_size=5):
    """
    Apply a median filter to the input data.
       
    :param data: 1D numpy array of data points
    :param window_size: Size of the filter window
    :return: Filtered data
    """
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    return np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])

def moving_average_filter(data, window_size=5):
    """
    Apply a moving average filter to the input data.
       
    :param data: 1D numpy array of data points
    :param window_size: Size of the filter window
    :return: Filtered data
    """
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    cumsum = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
