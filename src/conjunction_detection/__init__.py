"""
Conjunction Detection Module

This module provides efficient spatial indexing and conjunction detection
capabilities for satellite proximity analysis.
"""

from .spatial_index import SpatialIndex, KDTreeSpatialIndex
from .conjunction_analyzer import ConjunctionAnalyzer, ConjunctionEvent

__all__ = [
    'SpatialIndex',
    'KDTreeSpatialIndex',
    'ConjunctionAnalyzer',
    'ConjunctionEvent'
] 