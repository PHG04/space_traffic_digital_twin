"""
Sensor Simulation and Emulation Module

Provides realistic sensor network simulation with noise models,
measurement uncertainties, and sensor fusion capabilities.
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# Handle both relative and absolute imports
try:
    from .noise_models import (
        NoiseModel, NoiseCharacteristics, GaussianNoiseModel, 
        RadarNoiseModel, CorrelatedNoiseModel, MultiSensorNoiseModel
    )
    from ..orbital_mechanics.satellite_state import SatelliteState, StateVector
except ImportError:
    # If relative import fails, try absolute import for when running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from sensor_simulation.noise_models import (
        NoiseModel, NoiseCharacteristics, GaussianNoiseModel, 
        RadarNoiseModel, CorrelatedNoiseModel, MultiSensorNoiseModel
    )
    from orbital_mechanics.satellite_state import SatelliteState, StateVector


class SensorType(Enum):
    """Types of sensors for satellite tracking."""
    RADAR = "radar"
    OPTICAL = "optical"
    LASER = "laser"
    RADIO = "radio"
    GPS = "gps"


@dataclass
class SensorConfig:
    """Configuration for a sensor system."""
    
    sensor_id: str
    sensor_type: SensorType
    location: np.ndarray  # Earth-fixed coordinates (km)
    noise_characteristics: NoiseCharacteristics
    operational_range_km: float = 3000.0
    min_elevation_deg: float = 10.0
    max_tracking_rate_deg_s: float = 10.0
    measurement_interval_s: float = 1.0
    availability: float = 0.95  # Fraction of time sensor is available
    accuracy_degradation_factor: float = 1.0
    calibration_drift_rate: float = 0.001  # Per day
    
    # Sensor-specific parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SensorReading:
    """Represents a sensor measurement."""
    
    sensor_id: str
    satellite_id: str
    timestamp: datetime
    position: np.ndarray  # Measured position (km)
    velocity: np.ndarray  # Measured velocity (km/s)
    range_km: float
    azimuth_deg: float
    elevation_deg: float
    range_rate_km_s: float
    signal_strength: float = 1.0
    measurement_quality: float = 1.0
    uncertainty_position: np.ndarray = None
    uncertainty_velocity: np.ndarray = None
    
    def __post_init__(self):
        if self.uncertainty_position is None:
            self.uncertainty_position = np.ones(3) * 0.1  # Default 100m uncertainty
        if self.uncertainty_velocity is None:
            self.uncertainty_velocity = np.ones(3) * 0.001  # Default 1 m/s uncertainty


class SensorStation:
    """
    Individual sensor station with realistic tracking capabilities.
    """
    
    def __init__(self, config: SensorConfig, seed: int = None):
        """
        Initialize sensor station.
        
        Args:
            config: Sensor configuration
            seed: Random seed for noise generation
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize noise model based on sensor type
        self.noise_model = self._create_noise_model(seed)
        
        # State tracking
        self.is_operational = True
        self.last_calibration = datetime.now()
        self.calibration_drift = 0.0
        self.current_targets: Dict[str, datetime] = {}
        
        # Performance tracking
        self.measurement_count = 0
        self.successful_measurements = 0
        self.tracking_errors = 0
        
        # Random number generator for availability
        self.rng = np.random.RandomState(seed)
        
        self.logger.debug(f"Sensor station {config.sensor_id} initialized")
    
    def can_track_satellite(self, satellite_state: SatelliteState, timestamp: datetime) -> bool:
        """
        Determine if sensor can track the given satellite.
        
        Args:
            satellite_state: Satellite to track
            timestamp: Current time
            
        Returns:
            True if satellite can be tracked
        """
        if not self.is_operational or not satellite_state.state_vector:
            return False
        
        # Check availability (random outages)
        if self.rng.random() > self.config.availability:
            return False
        
        # Calculate look angles
        range_km, azimuth_deg, elevation_deg = self._calculate_look_angles(
            satellite_state.state_vector.position, timestamp
        )
        
        # Check constraints
        if range_km > self.config.operational_range_km:
            return False
        
        if elevation_deg < self.config.min_elevation_deg:
            return False
        
        # Check tracking rate limits (simplified)
        if satellite_state.satellite_id in self.current_targets:
            last_time = self.current_targets[satellite_state.satellite_id]
            time_diff = (timestamp - last_time).total_seconds()
            if time_diff < self.config.measurement_interval_s:
                return False
        
        return True
    
    def measure_satellite(self, satellite_state: SatelliteState, timestamp: datetime) -> Optional[SensorReading]:
        """
        Generate a sensor measurement for the satellite.
        
        Args:
            satellite_state: Satellite to measure
            timestamp: Measurement timestamp
            
        Returns:
            Sensor reading or None if measurement failed
        """
        if not self.can_track_satellite(satellite_state, timestamp):
            return None
        
        try:
            # Get true state
            true_position = satellite_state.state_vector.position
            true_velocity = satellite_state.state_vector.velocity
            
            # Apply sensor noise
            noisy_position, noisy_velocity = self.noise_model.apply_noise(
                true_position, true_velocity, timestamp
            )
            
            # Apply calibration drift
            drift_factor = self._get_calibration_drift(timestamp)
            noisy_position *= (1.0 + drift_factor)
            
            # Calculate spherical coordinates
            range_km, azimuth_deg, elevation_deg = self._calculate_look_angles(
                noisy_position, timestamp
            )
            
            # Calculate range rate
            relative_velocity = noisy_velocity - np.zeros(3)  # Sensor velocity (simplified)
            range_vector = noisy_position - self.config.location
            range_unit = range_vector / np.linalg.norm(range_vector)
            range_rate = np.dot(relative_velocity, range_unit)
            
            # Calculate signal strength and quality based on range and elevation
            signal_strength = self._calculate_signal_strength(range_km, elevation_deg)
            measurement_quality = self._calculate_measurement_quality(signal_strength, elevation_deg)
            
            # Calculate uncertainties
            uncertainty_pos = self._calculate_position_uncertainty(range_km, elevation_deg)
            uncertainty_vel = self._calculate_velocity_uncertainty(range_km, elevation_deg)
            
            # Create measurement
            reading = SensorReading(
                sensor_id=self.config.sensor_id,
                satellite_id=satellite_state.satellite_id,
                timestamp=timestamp,
                position=noisy_position,
                velocity=noisy_velocity,
                range_km=range_km,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                range_rate_km_s=range_rate,
                signal_strength=signal_strength,
                measurement_quality=measurement_quality,
                uncertainty_position=uncertainty_pos,
                uncertainty_velocity=uncertainty_vel
            )
            
            # Update tracking
            self.current_targets[satellite_state.satellite_id] = timestamp
            self.measurement_count += 1
            self.successful_measurements += 1
            
            return reading
            
        except Exception as e:
            self.logger.error(f"Error generating measurement: {e}")
            self.tracking_errors += 1
            return None
    
    def _create_noise_model(self, seed: int) -> NoiseModel:
        """Create appropriate noise model for sensor type."""
        if self.config.sensor_type == SensorType.RADAR:
            radar_config = self.config.additional_params.get('radar_config', {})
            return RadarNoiseModel(self.config.noise_characteristics, radar_config, seed)
        elif self.config.sensor_type == SensorType.OPTICAL:
            correlation_config = self.config.additional_params.get('correlation_config', {})
            return CorrelatedNoiseModel(self.config.noise_characteristics, correlation_config, seed)
        else:
            return GaussianNoiseModel(self.config.noise_characteristics, seed)
    
    def _calculate_look_angles(self, position: np.ndarray, timestamp: datetime) -> Tuple[float, float, float]:
        """Calculate range, azimuth, and elevation to satellite."""
        # Vector from sensor to satellite
        range_vector = position - self.config.location
        range_km = np.linalg.norm(range_vector)
        
        # Calculate spherical coordinates (simplified - assumes sensor at origin)
        x, y, z = range_vector
        azimuth_rad = np.arctan2(y, x)
        elevation_rad = np.arcsin(z / range_km)
        
        azimuth_deg = np.degrees(azimuth_rad)
        elevation_deg = np.degrees(elevation_rad)
        
        # Normalize azimuth to [0, 360)
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        return range_km, azimuth_deg, elevation_deg
    
    def _get_calibration_drift(self, timestamp: datetime) -> float:
        """Calculate current calibration drift."""
        days_since_calibration = (timestamp - self.last_calibration).total_seconds() / 86400.0
        return self.config.calibration_drift_rate * days_since_calibration
    
    def _calculate_signal_strength(self, range_km: float, elevation_deg: float) -> float:
        """Calculate signal strength based on range and elevation."""
        # Simple model: 1/r^2 falloff with elevation effects
        range_factor = (1000.0 / max(range_km, 100.0))**2
        elevation_factor = np.sin(np.radians(max(elevation_deg, 5.0)))
        
        signal_strength = range_factor * elevation_factor
        return min(1.0, max(0.0, signal_strength))
    
    def _calculate_measurement_quality(self, signal_strength: float, elevation_deg: float) -> float:
        """Calculate measurement quality metric."""
        # Quality degrades with low signal and low elevation
        signal_quality = signal_strength
        elevation_quality = min(1.0, elevation_deg / 45.0)  # Best at 45+ degrees
        
        overall_quality = (signal_quality * elevation_quality) * self.config.accuracy_degradation_factor
        return min(1.0, max(0.0, overall_quality))
    
    def _calculate_position_uncertainty(self, range_km: float, elevation_deg: float) -> np.ndarray:
        """Calculate position measurement uncertainty."""
        base_uncertainty = np.array([
            self.config.noise_characteristics.position_sigma,
            self.config.noise_characteristics.position_sigma,
            self.config.noise_characteristics.position_sigma
        ])
        
        # Scale with range and elevation
        range_factor = max(1.0, range_km / 1000.0)
        elevation_factor = max(1.0, 1.0 / np.sin(np.radians(max(elevation_deg, 10.0))))
        
        return base_uncertainty * range_factor * elevation_factor
    
    def _calculate_velocity_uncertainty(self, range_km: float, elevation_deg: float) -> np.ndarray:
        """Calculate velocity measurement uncertainty."""
        base_uncertainty = np.array([
            self.config.noise_characteristics.velocity_sigma,
            self.config.noise_characteristics.velocity_sigma,
            self.config.noise_characteristics.velocity_sigma
        ])
        
        # Scale with range and elevation
        range_factor = max(1.0, range_km / 1000.0)
        elevation_factor = max(1.0, 1.0 / np.sin(np.radians(max(elevation_deg, 10.0))))
        
        return base_uncertainty * range_factor * elevation_factor
    
    def perform_calibration(self, timestamp: datetime):
        """Perform sensor calibration."""
        self.last_calibration = timestamp
        self.calibration_drift = 0.0
        self.logger.info(f"Sensor {self.config.sensor_id} calibrated")
    
    def set_operational_status(self, operational: bool):
        """Set sensor operational status."""
        self.is_operational = operational
        if not operational:
            self.current_targets.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor performance statistics."""
        success_rate = (self.successful_measurements / max(1, self.measurement_count))
        
        return {
            'sensor_id': self.config.sensor_id,
            'sensor_type': self.config.sensor_type.value,
            'is_operational': self.is_operational,
            'measurement_count': self.measurement_count,
            'successful_measurements': self.successful_measurements,
            'tracking_errors': self.tracking_errors,
            'success_rate': success_rate,
            'active_targets': len(self.current_targets),
            'calibration_drift': self.calibration_drift,
            'days_since_calibration': (datetime.now() - self.last_calibration).total_seconds() / 86400.0
        }


class SensorEmulator:
    """
    Main sensor emulation system managing multiple sensor stations.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize sensor emulator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Sensor network
        self.sensor_stations: Dict[str, SensorStation] = {}
        
        # Measurement storage
        self.recent_measurements: Dict[str, List[SensorReading]] = {}
        self.measurement_history_limit = self.config.get('measurement_history_limit', 1000)
        
        # Fusion configuration
        self.enable_sensor_fusion = self.config.get('enable_sensor_fusion', True)
        self.fusion_time_window_s = self.config.get('fusion_time_window_s', 5.0)
        
        self.logger.info("Sensor Emulator initialized")
    
    def add_sensor_station(self, config: SensorConfig, seed: int = None) -> bool:
        """
        Add a sensor station to the network.
        
        Args:
            config: Sensor configuration
            seed: Random seed for sensor
            
        Returns:
            True if successful
        """
        if config.sensor_id in self.sensor_stations:
            self.logger.warning(f"Sensor {config.sensor_id} already exists")
            return False
        
        station = SensorStation(config, seed)
        self.sensor_stations[config.sensor_id] = station
        self.recent_measurements[config.sensor_id] = []
        
        self.logger.info(f"Added sensor station {config.sensor_id}")
        return True
    
    def remove_sensor_station(self, sensor_id: str) -> bool:
        """Remove a sensor station from the network."""
        if sensor_id not in self.sensor_stations:
            return False
        
        del self.sensor_stations[sensor_id]
        del self.recent_measurements[sensor_id]
        
        self.logger.info(f"Removed sensor station {sensor_id}")
        return True
    
    def measure_satellites(self, satellites: Dict[str, SatelliteState], 
                         timestamp: datetime) -> Dict[str, List[SensorReading]]:
        """
        Generate measurements for all satellites from all available sensors.
        
        Args:
            satellites: Dictionary of satellites to measure
            timestamp: Measurement timestamp
            
        Returns:
            Dictionary mapping sensor IDs to lists of measurements
        """
        all_measurements = {}
        
        for sensor_id, station in self.sensor_stations.items():
            measurements = []
            
            for satellite_id, satellite in satellites.items():
                reading = station.measure_satellite(satellite, timestamp)
                if reading:
                    measurements.append(reading)
            
            all_measurements[sensor_id] = measurements
            
            # Update measurement history
            self.recent_measurements[sensor_id].extend(measurements)
            
            # Limit history size
            if len(self.recent_measurements[sensor_id]) > self.measurement_history_limit:
                excess = len(self.recent_measurements[sensor_id]) - self.measurement_history_limit
                self.recent_measurements[sensor_id] = self.recent_measurements[sensor_id][excess:]
        
        return all_measurements
    
    def get_fused_measurements(self, satellite_id: str, 
                             timestamp: datetime,
                             time_window_s: float = None) -> Optional[SensorReading]:
        """
        Get fused measurements for a satellite from multiple sensors.
        
        Args:
            satellite_id: ID of satellite
            timestamp: Target timestamp
            time_window_s: Time window for fusion (default: config value)
            
        Returns:
            Fused sensor reading or None
        """
        if not self.enable_sensor_fusion:
            return None
        
        if time_window_s is None:
            time_window_s = self.fusion_time_window_s
        
        # Collect relevant measurements
        relevant_measurements = []
        
        for sensor_id, measurements in self.recent_measurements.items():
            for measurement in measurements:
                if (measurement.satellite_id == satellite_id and
                    abs((measurement.timestamp - timestamp).total_seconds()) <= time_window_s):
                    relevant_measurements.append(measurement)
        
        if len(relevant_measurements) < 2:
            # Need at least 2 measurements for fusion
            return relevant_measurements[0] if relevant_measurements else None
        
        # Simple weighted fusion based on measurement quality
        total_weight = 0.0
        fused_position = np.zeros(3)
        fused_velocity = np.zeros(3)
        fused_uncertainties_pos = np.zeros(3)
        fused_uncertainties_vel = np.zeros(3)
        
        best_measurement = None
        best_quality = 0.0
        
        for measurement in relevant_measurements:
            weight = measurement.measurement_quality * measurement.signal_strength
            total_weight += weight
            
            fused_position += weight * measurement.position
            fused_velocity += weight * measurement.velocity
            
            # Track best measurement for metadata
            if measurement.measurement_quality > best_quality:
                best_quality = measurement.measurement_quality
                best_measurement = measurement
        
        if total_weight > 0:
            fused_position /= total_weight
            fused_velocity /= total_weight
            
            # Calculate fused uncertainties (simplified)
            for measurement in relevant_measurements:
                weight = measurement.measurement_quality * measurement.signal_strength / total_weight
                fused_uncertainties_pos += weight * measurement.uncertainty_position**2
                fused_uncertainties_vel += weight * measurement.uncertainty_velocity**2
            
            fused_uncertainties_pos = np.sqrt(fused_uncertainties_pos)
            fused_uncertainties_vel = np.sqrt(fused_uncertainties_vel)
            
            # Create fused measurement using best measurement as template
            fused_reading = SensorReading(
                sensor_id="FUSED",
                satellite_id=satellite_id,
                timestamp=timestamp,
                position=fused_position,
                velocity=fused_velocity,
                range_km=np.linalg.norm(fused_position),
                azimuth_deg=0.0,  # Not meaningful for fused measurement
                elevation_deg=0.0,  # Not meaningful for fused measurement
                range_rate_km_s=0.0,  # Not meaningful for fused measurement
                signal_strength=np.mean([m.signal_strength for m in relevant_measurements]),
                measurement_quality=np.mean([m.measurement_quality for m in relevant_measurements]),
                uncertainty_position=fused_uncertainties_pos,
                uncertainty_velocity=fused_uncertainties_vel
            )
            
            return fused_reading
        
        return None
    
    def get_sensor_coverage(self, timestamp: datetime) -> Dict[str, Dict]:
        """Get coverage information for all sensors."""
        coverage = {}
        
        for sensor_id, station in self.sensor_stations.items():
            coverage[sensor_id] = {
                'operational': station.is_operational,
                'location': station.config.location.tolist(),
                'range_km': station.config.operational_range_km,
                'min_elevation_deg': station.config.min_elevation_deg,
                'availability': station.config.availability,
                'active_targets': len(station.current_targets)
            }
        
        return coverage
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        total_measurements = sum(station.measurement_count for station in self.sensor_stations.values())
        total_successful = sum(station.successful_measurements for station in self.sensor_stations.values())
        total_errors = sum(station.tracking_errors for station in self.sensor_stations.values())
        
        sensor_stats = {}
        for sensor_id, station in self.sensor_stations.items():
            sensor_stats[sensor_id] = station.get_statistics()
        
        return {
            'total_sensors': len(self.sensor_stations),
            'operational_sensors': sum(1 for s in self.sensor_stations.values() if s.is_operational),
            'total_measurements': total_measurements,
            'successful_measurements': total_successful,
            'total_errors': total_errors,
            'overall_success_rate': total_successful / max(1, total_measurements),
            'sensor_fusion_enabled': self.enable_sensor_fusion,
            'individual_sensor_stats': sensor_stats
        } 