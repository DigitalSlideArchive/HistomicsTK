# -*- coding: utf-8 -*-

# from .BinaryMixtureCut import BinaryMixtureCut
from .ChanVese import ChanVese
from .cLoG import cLoG
from .ColorConvolution import ColorConvolution
from .ColorDeconvolution import ColorDeconvolution
from .ComplementStainMatrix import ComplementStainMatrix
from .CondenseLabel import CondenseLabel
from .ConvertSchedule import ConvertSchedule
from .Del2 import Del2
from .DregEdge import DregEdge
from .EmbedBounds import EmbedBounds
from .EstimateVariance import EstimateVariance
from .FeatureExtraction import FeatureExtraction
from .FilterLabel import FilterLabel
from .GaussianGradient import GaussianGradient
from .GaussianVoting import GaussianVoting
from .gLoG import gLoG
from .GradientDiffusion import GradientDiffusion
from .GradientFlow import GradientFlow
from .GraphColorSequential import GraphColorSequential
from .LabelPerimeter import LabelPerimeter
from .LabelRegionAdjacency import LabelRegionAdjacency
from .MaxClustering import MaxClustering
from .MergeColinear import MergeColinear
from .MergeSinks import MergeSinks
from .OpticalDensityFwd import OpticalDensityFwd
from .OpticalDensityInv import OpticalDensityInv
from .PoissonMixture import PoissonMixture
from .RegionAdjacencyLayer import RegionAdjacencyLayer
from .ReinhardNorm import ReinhardNorm
from .ReinhardSample import ReinhardSample
from .RudermanLABFwd import RudermanLABFwd
from .RudermanLABInv import RudermanLABInv
from .Sample import Sample
from .ShuffleLabel import ShuffleLabel
from .SimpleMask import SimpleMask
from .SparseColorDeconvolution import SparseColorDeconvolution
from .SubmitTorque import SubmitTorque
from .TilingSchedule import TilingSchedule

__version__ = '0.1.0'

__all__ = (  # 'BinaryMixtureCut',  # disable until pygco is deployed
           'ChanVese',
           'cLoG',
           'ColorConvolution',
           'ColorDeconvolution',
           'ComplementStainMatrix',
           'CondenseLabel',
           'ConvertSchedule',
           'Del2',
           'DregEdge',
           'EmbedBounds',
           'EstimateVariance',
           'FeatureExtraction',
           'FilterLabel',
           'GaussianGradient',
           'GaussianVoting',
           'GraphColorSequential',
           'gLoG',
           'GradientDiffusion',
           'GradientFlow',
           'LabelPerimeter',
           'LabelRegionAdjacency',
           'MaxClustering',
           'MergeColinear',
           'MergeSinks',
           'OpticalDensityFwd',
           'OpticalDensityInv',
           'PoissonMixture',
           'RegionAdjacencyLayer',
           'ReinhardNorm',
           'ReinhardSample',
           'RudermanLABFwd',
           'RudermanLABInv',
           'Sample',
           'ShuffleLabel',
           'SimpleMask',
           'SparseColorDeconvolution',
           'SubmitTorque',
           'TilingSchedule')
