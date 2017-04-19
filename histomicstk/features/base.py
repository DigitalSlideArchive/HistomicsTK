import abc
import six


class BaseFeatures(six.with_metaclass(abc.ABCMeta, object)):
    """Abstract base class that defines the API that all feature extraction
    classes must implement

    Parameters
    -----------
    im_label: array_like
        Labeled mask image. Pixel intensity represents the ID of the object it
         belongs to. Non-zero values are considered to be foreground objects.

    im_input : array_like or None
        Input intensity image with same size as the labeled mask.

    feature_names : list of str or None
        Names of the desired list of features to compute.
    """

    # Dictionary of features wherein
    #   key = feature name,
    #   value = True/False indicating if it should be computed by default
    _feature_dict = {}

    def __init__(self, im_label, im_input=None,
                 feature_names=None, *args, **kwargs):

        self._im_label = im_label
        self._im_input = im_input

        # If feature names are not specified a default will be computed
        self._default_features = [fname
                                  for fname, flag in self._feature_dict.items()
                                  if flag is True]

        if feature_names is None:
            self._features = self.get_default_feature_names()
        else:
            # check if the specified list of features are valid
            for fname in feature_names:
                if fname not in self._feature_dict:
                    raise ValueError('Invalid feature name %s')

            self._features_names = feature_names


    @classmethod
    def get_all_feature_names(cls):
        """Returns the list of all features that this class can compute"""

        return cls._feature_dict.keys()

    @classmethod
    def get_default_feature_names(cls):
        """Returns the list of features that this class computes"""

        default_features = [fname
                            for fname, flag in self._feature_dict.items()
                            if flag is True]
        return default_features

    def get_feature_names(self):
        """Returns the list of features that this class computes"""
        return self._feature_names

    @abc.abstractmethod
    def get_features(self):
        """computes the desired set of features for each foreground object
        and returns them in the form of a pandas data frame
        """
        return
