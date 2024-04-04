"""
Created on Tue Sep 24 00:43:04 2019.

@author: mtageld
"""
import datetime
import logging
import os


class Print_and_log:
    """Print to screen and/or log if conditions are satisfied (Internal)."""

    def __init__(self, verbose=True, logger=None):
        """Init. This is for PEP compliance."""
        self.verbose = verbose
        self.keep_log = logger is not None
        if self.keep_log:
            self.logger = logger

    def _print(self, text):
        if self.verbose:
            print(text)
        if self.keep_log:
            self.logger.info(text)


class Base_HTK_Class:
    """Just a base class with preferred behavior to inherit."""

    def __init__(self, default_attr=None, more_allowed_attr=None, **kwargs):
        """Init base HTK class."""
        default_attr = {} if default_attr is None else default_attr
        more_allowed_attr = [] if more_allowed_attr is None else more_allowed_attr

        da = {
            'verbose': 1,
            'monitorPrefix': '',
            'logging_savepath': None,
            'suppress_warnings': False,
        }
        default_attr.update(
            (k, v) for k, v in da.items() if k not in default_attr.keys())

        # See: https://stackoverflow.com/questions/8187082/how-can-you-set-...
        # class-attributes-from-variable-arguments-kwargs-in-python
        allowed_attr = list(default_attr.keys()) + more_allowed_attr
        default_attr.update(kwargs)
        self.__dict__.update(
            (k, v) for k, v in default_attr.items() if k in allowed_attr)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_attr)
        if rejected_keys:
            msg = f'Invalid arguments in constructor:{rejected_keys}'
            raise ValueError(
                msg)

        # configure logger
        self.keep_log = self.logging_savepath is not None
        if self.keep_log:
            logger = logging.getLogger()
            self.logname = os.path.join(
                self.logging_savepath,
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.log')
            logging.basicConfig(filename=self.logname, level=logging.INFO)
        else:
            logger = None

        # verbosity control
        self.cpr1 = Print_and_log(verbose=self.verbose >= 1, logger=logger)
        self._print1 = self.cpr1._print
        self.cpr2 = Print_and_log(verbose=self.verbose >= 2, logger=logger)
        self._print2 = self.cpr2._print
        if self.keep_log:
            self._print1('Saving logs to: %s' % self.logname)
