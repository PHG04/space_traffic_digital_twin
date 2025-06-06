"""
Noise Models for Sensor Data Simulation

Implements various noise models for realistic sensor data emulation,
including Gaussian noise, radar-specific errors, and correlated noise.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class NoiseCharacteristics:
    """Characteristics of sensor noise."""
    
    position_sigma: float  # Position noise standard deviation (km)
    velocity_sigma: float  # Velocity noise standard deviation (km/s)
    range_sigma: float     # Range measurement noise (km)
    azimuth_sigma: float   # Azimuth noise (degrees)
    elevation_sigma: float # Elevation noise (degrees)
    correlation_time: float = 300.0  # Correlation time (seconds)
    bias_stability: float = 0.001    # Bias stability (km)


class NoiseModel(ABC):
    """Abstract base class for sensor noise models."""
    
    @abstractmethod
    def apply_noise(self, 
                   position: np.ndarray, 
                   velocity: np.ndarray,
                   timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply noise to position and velocity measurements.
        
        Args:
            position: True position vector (km)
            velocity: True velocity vector (km/s)
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (noisy_position, noisy_velocity)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal noise state."""
        pass


class GaussianNoiseModel(NoiseModel):
    """
    Simple Gaussian white noise model for sensor measurements.
    """
    
    def __init__(self, characteristics: NoiseCharacteristics, seed: int = None):
        """
        Initialize Gaussian noise model.
        
        Args:
            characteristics: Noise characteristics
            seed: Random seed for reproducibility
        """
        self.characteristics = characteristics
        self.logger = logging.getLogger(__name__)
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        self.logger.debug("Gaussian noise model initialized")
    
    def apply_noise(self, 
                   position: np.ndarray, 
                   velocity: np.ndarray,
                   timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Apply independent Gaussian noise to measurements."""
        
        # Position noise (3D Gaussian)
        position_noise = self.rng.normal(
            0.0, 
            self.characteristics.position_sigma, 
            size=3
        )
        
        # Velocity noise (3D Gaussian)
        velocity_noise = self.rng.normal(
            0.0, 
            self.characteristics.velocity_sigma, 
            size=3
        )
        
        # Apply noise
        noisy_position = position + position_noise
        noisy_velocity = velocity + velocity_noise
        
        return noisy_position, noisy_velocity
    
    def reset(self):
        """Reset random generator state."""
        # No internal state to reset for white noise
        pass


class RadarNoiseModel(NoiseModel):
    """
    Realistic radar tracking noise model with range-dependent errors.
    """
    
    def __init__(self, characteristics: NoiseCharacteristics, 
                 radar_config: Dict = None, seed: int = None):
        """
        Initialize radar noise model.
        
        Args:
            characteristics: Base noise characteristics
            radar_config: Radar-specific configuration
            seed: Random seed
        """
        self.characteristics = characteristics
        self.radar_config = radar_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Radar parameters
        self.min_elevation = self.radar_config.get('min_elevation_deg', 10.0)
        self.max_range = self.radar_config.get('max_range_km', 3000.0)
        self.range_bias = self.radar_config.get('range_bias_km', 0.0)
        self.azimuth_bias = self.radar_config.get('azimuth_bias_deg', 0.0)
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Internal state for correlated errors
        self.range_bias_state = 0.0
        self.azimuth_bias_state = 0.0
        self.last_update_time = None
        
        self.logger.debug("Radar noise model initialized")
    
    def apply_noise(self, 
                   position: np.ndarray, 
                   velocity: np.ndarray,
                   timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Apply radar-specific noise model."""
        
        # Calculate range-dependent noise scaling
        range_km = np.linalg.norm(position)
        elevation_rad = np.arcsin(position[2] / range_km)
        elevation_deg = np.degrees(elevation_rad)
        
        # Range-dependent noise scaling
        range_factor = max(1.0, range_km / 1000.0)  # Increase noise with range
        elevation_factor = max(1.0, 1.0 / np.sin(np.radians(max(elevation_deg, 5.0))))
        
        # Update correlated biases
        self._update_correlated_biases(timestamp)
        
        # Convert to spherical coordinates for realistic radar errors
        range_meas, azimuth_rad, elevation_rad = self._cartesian_to_spherical(position)
        range_rate = np.dot(position, velocity) / range_km
        
        # Apply noise to spherical measurements
        noisy_range = range_meas + self.rng.normal(
            self.range_bias_state, 
            self.characteristics.range_sigma * range_factor
        )
        
        noisy_azimuth = azimuth_rad + np.radians(self.rng.normal(
            self.azimuth_bias_state,
            self.characteristics.azimuth_sigma * elevation_factor
        ))
        
        noisy_elevation = elevation_rad + np.radians(self.rng.normal(
            0.0,
            self.characteristics.elevation_sigma * elevation_factor
        ))
        
        noisy_range_rate = range_rate + self.rng.normal(
            0.0,
            self.characteristics.velocity_sigma * range_factor
        )
        
        # Convert back to Cartesian coordinates
        noisy_position = self._spherical_to_cartesian(
            noisy_range, noisy_azimuth, noisy_elevation
        )
        
        # Approximate velocity from noisy measurements
        # (In practice, would use more sophisticated tracking filters)
        velocity_direction = velocity / np.linalg.norm(velocity)
        noisy_velocity = velocity_direction * noisy_range_rate
        
        # Add cross-track velocity noise
        cross_track_noise = self.rng.normal(
            0.0, 
            self.characteristics.velocity_sigma, 
            size=3
        )
        cross_track_noise = cross_track_noise - np.dot(cross_track_noise, velocity_direction) * velocity_direction
        noisy_velocity += cross_track_noise
        
        return noisy_position, noisy_velocity
    
    def _update_correlated_biases(self, timestamp: datetime):
        """Update slowly varying bias terms."""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return
        
        dt = (timestamp - self.last_update_time).total_seconds()
        self.last_update_time = timestamp
        
        # Exponential correlation model
        correlation_factor = np.exp(-dt / self.characteristics.correlation_time)
        
        # Update range bias
        self.range_bias_state = (
            self.range_bias_state * correlation_factor + 
            self.rng.normal(0.0, self.characteristics.bias_stability * np.sqrt(1 - correlation_factor**2))
        )
        
        # Update azimuth bias
        self.azimuth_bias_state = (
            self.azimuth_bias_state * correlation_factor +
            self.rng.normal(0.0, 0.1 * np.sqrt(1 - correlation_factor**2))  # 0.1 degree bias variation
        )
    
    def _cartesian_to_spherical(self, position: np.ndarray) -> Tuple[float, float, float]:
        """Convert Cartesian to spherical coordinates."""
        x, y, z = position
        
        range_val = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / range_val)
        
        return range_val, azimuth, elevation
    
    def _spherical_to_cartesian(self, range_val: float, azimuth: float, elevation: float) -> np.ndarray:
        """Convert spherical to Cartesian coordinates."""
        x = range_val * np.cos(elevation) * np.cos(azimuth)
        y = range_val * np.cos(elevation) * np.sin(azimuth)
        z = range_val * np.sin(elevation)
        
        return np.array([x, y, z])
    
    def reset(self):
        """Reset internal bias states."""
        self.range_bias_state = 0.0
        self.azimuth_bias_state = 0.0
        self.last_update_time = None


class CorrelatedNoiseModel(NoiseModel):
    """
    Correlated noise model with temporal and spatial correlations.
    """
    
    def __init__(self, characteristics: NoiseCharacteristics, 
                 correlation_config: Dict = None, seed: int = None):
        """
        Initialize correlated noise model.
        
        Args:
            characteristics: Noise characteristics
            correlation_config: Correlation configuration
            seed: Random seed
        """
        self.characteristics = characteristics
        self.correlation_config = correlation_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Correlation parameters
        self.temporal_correlation = self.correlation_config.get('temporal_correlation', 0.9)
        self.spatial_correlation = self.correlation_config.get('spatial_correlation', 0.5)
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # State for correlated noise
        self.previous_position_noise = np.zeros(3)
        self.previous_velocity_noise = np.zeros(3)
        self.last_update_time = None
        
        self.logger.debug("Correlated noise model initialized")
    
    def apply_noise(self, 
                   position: np.ndarray, 
                   velocity: np.ndarray,
                   timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporally and spatially correlated noise."""
        
        # Calculate time step
        if self.last_update_time is not None:
            dt = (timestamp - self.last_update_time).total_seconds()
            correlation_factor = self.temporal_correlation ** (dt / 60.0)  # Decorrelation over minutes
        else:
            correlation_factor = 0.0
        
        self.last_update_time = timestamp
        
        # Generate new uncorrelated noise
        new_position_noise = self.rng.normal(
            0.0, 
            self.characteristics.position_sigma, 
            size=3
        )
        
        new_velocity_noise = self.rng.normal(
            0.0, 
            self.characteristics.velocity_sigma, 
            size=3
        )
        
        # Apply temporal correlation
        position_noise = (
            self.previous_position_noise * correlation_factor +
            new_position_noise * np.sqrt(1 - correlation_factor**2)
        )
        
        velocity_noise = (
            self.previous_velocity_noise * correlation_factor +
            new_velocity_noise * np.sqrt(1 - correlation_factor**2)
        )
        
        # Apply spatial correlation between position and velocity
        if self.spatial_correlation > 0:
            # Add cross-correlation
            position_to_velocity = (
                position_noise * self.spatial_correlation * 
                (self.characteristics.velocity_sigma / self.characteristics.position_sigma)
            )
            velocity_noise += position_to_velocity
        
        # Update state
        self.previous_position_noise = position_noise
        self.previous_velocity_noise = velocity_noise
        
        # Apply noise
        noisy_position = position + position_noise
        noisy_velocity = velocity + velocity_noise
        
        return noisy_position, noisy_velocity
    
    def reset(self):
        """Reset correlation state."""
        self.previous_position_noise = np.zeros(3)
        self.previous_velocity_noise = np.zeros(3)
        self.last_update_time = None


class MultiSensorNoiseModel(NoiseModel):
    """
    Multi-sensor fusion noise model combining different sensor types.
    """
    
    def __init__(self, sensor_configs: Dict[str, Dict], seed: int = None):
        """
        Initialize multi-sensor noise model.
        
        Args:
            sensor_configs: Dictionary of sensor configurations
            seed: Random seed
        """
        self.sensor_configs = sensor_configs
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual sensor models
        self.sensor_models = {}
        base_seed = seed or 42
        
        for sensor_name, config in sensor_configs.items():
            characteristics = NoiseCharacteristics(**config.get('characteristics', {}))
            sensor_type = config.get('type', 'gaussian')
            
            if sensor_type == 'gaussian':
                model = GaussianNoiseModel(characteristics, base_seed + hash(sensor_name) % 1000)
            elif sensor_type == 'radar':
                model = RadarNoiseModel(characteristics, config.get('radar_config', {}), 
                                      base_seed + hash(sensor_name) % 1000)
            elif sensor_type == 'correlated':
                model = CorrelatedNoiseModel(characteristics, config.get('correlation_config', {}),
                                           base_seed + hash(sensor_name) % 1000)
            else:
                model = GaussianNoiseModel(characteristics, base_seed + hash(sensor_name) % 1000)
            
            self.sensor_models[sensor_name] = model
        
        self.logger.debug(f"Multi-sensor noise model initialized with {len(self.sensor_models)} sensors")
    
    def apply_noise(self, 
                   position: np.ndarray, 
                   velocity: np.ndarray,
                   timestamp: datetime,
                   active_sensors: list = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply noise from multiple sensors.
        
        Returns:
            Dictionary mapping sensor names to (noisy_position, noisy_velocity) tuples
        """
        if active_sensors is None:
            active_sensors = list(self.sensor_models.keys())
        
        results = {}
        
        for sensor_name in active_sensors:
            if sensor_name in self.sensor_models:
                noisy_pos, noisy_vel = self.sensor_models[sensor_name].apply_noise(
                    position, velocity, timestamp
                )
                results[sensor_name] = (noisy_pos, noisy_vel)
        
        return results
    
    def reset(self):
        """Reset all sensor models."""
        for model in self.sensor_models.values():
            model.reset()
    
    def get_fused_measurement(self, 
                            sensor_measurements: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            weights: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse measurements from multiple sensors using weighted averaging.
        
        Args:
            sensor_measurements: Dictionary of sensor measurements
            weights: Optional weights for each sensor
            
        Returns:
            Fused (position, velocity) measurements
        """
        if not sensor_measurements:
            raise ValueError("No sensor measurements provided")
        
        if weights is None:
            # Equal weights
            weights = {name: 1.0 for name in sensor_measurements.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {name: w/total_weight for name, w in weights.items()}
        
        # Weighted fusion
        fused_position = np.zeros(3)
        fused_velocity = np.zeros(3)
        
        for sensor_name, (pos, vel) in sensor_measurements.items():
            weight = normalized_weights.get(sensor_name, 0.0)
            fused_position += weight * pos
            fused_velocity += weight * vel
        
        return fused_position, fused_velocity 