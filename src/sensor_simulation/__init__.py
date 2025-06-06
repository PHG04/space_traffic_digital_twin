"""
Sensor Simulation Module

This module provides realistic sensor simulation capabilities including
noise models, measurement uncertainties, and sensor network emulation.
"""

from .noise_models import (
    NoiseModel, NoiseCharacteristics, GaussianNoiseModel,
    RadarNoiseModel, CorrelatedNoiseModel, MultiSensorNoiseModel
)
from .sensor_emulator import SensorEmulator, SensorConfig, SensorReading, SensorType

__all__ = [
    'NoiseModel',
    'NoiseCharacteristics', 
    'GaussianNoiseModel',
    'RadarNoiseModel',
    'CorrelatedNoiseModel',
    'MultiSensorNoiseModel',
    'SensorEmulator',
    'SensorConfig',
    'SensorReading',
    'SensorType'
] 