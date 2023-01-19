import numpy as np
import cv2
import skimage.measure


def snr(image, ddof=0):
    """snr calculates the signal-to-noise ratio of an image.
    Input:  image
            ddof (default is 0)
    Output: snr of the image
    """
    array = np.asanyarray(image)
    array = (array - array.min()) / (array.max() - array.min())
    m = array.mean()
    std = array.std(ddof=ddof)

    return m, std


def my_cpp(image):
    """cpp is evaluating the contrast per pixel.
    Input:  image
    Output: cpp of the image
    """
    kernel = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8
    filtered_image = cv2.filter2D(image, -1, kernel)
    return np.mean(filtered_image)


def entropy(image, _min, _max, is_CT):
    """entropy calculates the entropy of a given image.
    Input:  Image
    Output: Entropy of the image
    """
    if is_CT:
        _min = -400
        _max = 600
    array = np.round(255*(np.asarray(image) - _min) / (_max - _min)).astype(int)
    return skimage.measure.shannon_entropy(image)


def statis_info(image, _min, _max, is_CT):
    """statis_info calculates the mean and the std of the given image.
    """
    if is_CT:
        _min = -400
        _max = 600
    array = (np.asarray(image) - _min) / (_max - _min)
    mean = np.mean(array)
    std = np.std(array)
    return mean, std
    