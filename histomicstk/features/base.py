import abc
import six


class FeaturesBase(six.with_metaclass(abc.ABCMeta, object)):
    """Abstract base class that defines the API that all feature extraction
    classes must implement

    Parameters
    -----------
    im_label: array_like
        Labeled mask image. Pixel intensity represents the ID of the object it
         belongs to. Non-zero values are considered to be foreground objects.

    im_input : array_like or None
        Input intensity image with same size as the labeled mask.

    features : list of str or None
        Names of the desired list of features to compute.
    """

    # Dictionary of features wherein
    #   key = feature name,
    #   value = True/False indicating if it should be computed or not
    __feature_dict = {}

    # List of feature names computed by default
    __default_features = None

    def __init__(self, im_label, im_input=None, features=None, *args, **kwargs):

        self._im_label = im_label
        self._im_input = im_input

        # If __default_features is not defined all features will be computed
        if self.__default_features:
            self.__default_features = self.__feature_dict.keys()

        # If feature names are not specified a default will be computed
        if features is None:
            features = self.__default_features

        # check if the specified list of features are valid
        for fname in features:
            if fname not in self.__feature_dict:
                raise ValueError('Invalid feature name %s')

        self._features = features


    def get_all_feature_names(self):
        """Returns the list of all features that this class can compute"""
        return

    def get_default_feature_names(self):
        """Returns the list of features that this class computes"""
        return

    def get_feature_names(self):
        """Returns the list of features that this class computes"""
        return

    @abc.abstractmethod
    def get_features(self):
        """computes the desired set of features for each foreground object
        and returns them in the form of a pandas data frame
        """
        return
