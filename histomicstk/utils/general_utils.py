#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:43:04 2019.

@author: mtageld
"""

# %% =====================================================================


class Conditional_Print(object):
    """Print to screen if certain conditions are satisfied (Internal)."""

    def __init__(self, verbose=True):
        """Init. This is for PEP compliance."""
        self.verbose = verbose

    def _print(self, text):
        if self.verbose:
            print(text)

# %% =====================================================================


class Base_HTK_Class(object):
    """Just a base class with preferred behavior to inherit."""

    def __init__(self, default_attr={}, more_allowed_attr=[], **kwargs):
        """Init base HTK class."""
        # see: https://stackoverflow.com/questions/8187082/how-can-you-set-...
        # class-attributes-from-variable-arguments-kwargs-in-python
        allowed_attr = list(default_attr.keys()) + more_allowed_attr
        default_attr.update(kwargs)
        self.__dict__.update(
            (k, v) for k, v in default_attr.items() if k in allowed_attr)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_attr)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor:{}".format(rejected_keys))

        # verbosity control
        self.cpr1 = Conditional_Print(verbose=self.verbose == 1)
        self._print1 = self.cpr1._print
        self.cpr2 = Conditional_Print(verbose=self.verbose == 2)
        self._print2 = self.cpr2._print


# %% =====================================================================
