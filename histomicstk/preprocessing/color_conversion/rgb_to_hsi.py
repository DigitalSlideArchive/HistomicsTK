"""Placeholder."""

import numpy as np


def rgb_to_hsi(im):
    """Convert to HSI the RGB pixels in im.

    Adapted from
    https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma.

    """
    im = np.moveaxis(im, -1, 0)
    if len(im) not in (3, 4):
        msg = (
            'Expected 3-channel RGB or 4-channel RGBA image;'
            ' received a {}-channel image'.format(len(im))
        )
        raise ValueError(msg)
    im = im[:3]
    hues = (np.arctan2(3**0.5 * (im[1] - im[2]),
                       2 * im[0] - im[1] - im[2]) / (2 * np.pi)) % 1
    intensities = im.mean(0)
    saturations = np.where(
        intensities, 1 - im.min(0) / np.maximum(intensities, 1e-10), 0)
    return np.stack([hues, saturations, intensities], -1)
