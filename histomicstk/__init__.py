# -*- coding: utf-8 -*-

# from .BinaryMixtureCut import BinaryMixtureCut
from .AreaOpenLabel import AreaOpenLabel
from .ChanVese import ChanVese
from .cLoG import cLoG
from .ColorConvolution import ColorConvolution
from .ColorDeconvolution import ColorDeconvolution
from .CompactLabel import CompactLabel
from .ComplementStainMatrix import ComplementStainMatrix
from .CondenseLabel import CondenseLabel
from .ConvertSchedule import ConvertSchedule
from .Del2 import Del2
from .DregEdge import DregEdge
from .Eigenvalues import Eigenvalues
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
from .Hessian import Hessian
from .LabelPerimeter import LabelPerimeter
from .LabelRegionAdjacency import LabelRegionAdjacency
from .MaxClustering import MaxClustering
from .MergeColinear import MergeColinear
from .MergeSinks import MergeSinks
from .MinimumModel import MinimumModel
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
from .SplitLabel import SplitLabel
from .SubmitTorque import SubmitTorque
from .TilingSchedule import TilingSchedule
from .TraceBounds import TraceBounds
from .Vesselness import Vesselness
from .WidthOpenLabel import WidthOpenLabel

__version__ = '0.1.0'

# Add 'BinaryMixtureCut' after pygco is deployed
__all__ = ('AreaOpenLabel',
		   'ChanVese',
           'cLoG',
           'ColorConvolution',
           'ColorDeconvolution',
           'CompactLabel',
           'ComplementStainMatrix',
           'CondenseLabel',
           'ConvertSchedule',
           'Del2',
           'DregEdge',
           'Eigenvalues',
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
           'Hessian',
           'LabelPerimeter',
           'LabelRegionAdjacency',
           'MaxClustering',
           'MergeColinear',
           'MergeSinks',
           'MinimumModel',
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
           'SplitLabel'
           'SubmitTorque',
           'TilingSchedule',
           'TraceBounds'
           'Vesselness'
           'WidthOpenLabel')
