# -*- coding: utf-8 -*-
"""
This package contains utility functions to augment images, which is most
relevant in training convolutional neural network models. See

Tellez, David, Geert Litjens, Peter Bandi, Wouter Bulten, John-Melle Bokhorst,
Francesco Ciompi, and Jeroen van der Laak. "Quantifying the effects of data
augmentation and stain color normalization in convolutional neural networks
for computational pathology." arXiv preprint arXiv:1902.06543 (2019).
"""
# make functions available at the package level using these shadow imports
# since we mostly have one function per file
from .color_augmentation import augment_stain_concentration

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'augment_stain_concentration',
)
