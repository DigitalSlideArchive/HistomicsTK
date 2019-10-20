# -*- coding: utf-8 -*-
"""
This package contains utility functions to augment images.

This is relevant in training convolutional neural network models. Refer to this study ...

Tellez, David, Geert Litjens, Peter Bandi, Wouter Bulten, John-Melle Bokhorst,
Francesco Ciompi, and Jeroen van der Laak. "Quantifying the effects of data
augmentation and stain color normalization in convolutional neural networks
for computational pathology." arXiv preprint arXiv:1902.06543 (2019).
"""
# make functions available at the package level using these shadow imports
# since we mostly have one function per file
from .color_augmentation import perturb_stain_concentration
from .color_augmentation import rgb_perturb_stain_concentration

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'perturb_stain_concentration',
    'rgb_perturb_stain_concentration',
)
