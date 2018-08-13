import numpy as np


def bilinear_upsampling_weights(factor, num_channels):
    """create filter for bilinear upsampling.
    reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/
    upsampling.py

    Parameters
    ----------
    factor : int
        factor by which upsampling should be performed
    num_channels : int
        number of channels in the image to upsample

    Returns
    ----------
    weights : numpy array
        array of shape [filter_size, filter_size, num_channels, num_channels) that contains weights for initializing a
        bilinear upsampling filter for conv2d_transpose
    """

    filter_size = 2 * factor - factor % 2

    weights = np.zeros((filter_size, filter_size, num_channels, num_channels),
                       dtype=np.float32)

    scale_factor = (filter_size + 1) // 2

    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5

    og = np.ogrid[:filter_size, :filter_size]

    kernel_values = (1 - abs(og[0] - center) / scale_factor) * \
        (1 - abs(og[1] - center) / scale_factor)

    for i in range(num_channels):
        weights[:, :, i, i] = kernel_values

    return weights
