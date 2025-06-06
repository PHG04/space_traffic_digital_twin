"""
Satellite State Management

Classes for managing satellite orbital states, elements, and state vectors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation


@dataclass
class OrbitalElements:
    """Keplerian orbital elements for a satellite."""
    
    # Classical orbital elements
    semi_major_axis: float  # km
    eccentricity: float     # dimensionless
    inclination: float      # degrees
    raan: float            # Right Ascension of Ascending Node, degrees
    arg_periapsis: float   # Argument of periapsis, degrees
    true_anomaly: float    # degrees
    
    # Reference epoch
    epoch: datetime
    
    def __post_init__(self):
        """Validate orbital elements."""
        if self.semi_major_axis <= 0:
            raise ValueError("Semi-major axis must be positive")
        if not 0 <= self.eccentricity < 1:
            raise ValueError("Eccentricity must be between 0 and 1 for elliptical orbits")
        if not 0 <= self.inclination <= 180:
            raise ValueError("Inclination must be between 0 and 180 degrees")
    
    @property
    def period(self) -> float:
        """Calculate orbital period in seconds."""
        mu = 398600.4418  # Earth's gravitational parameter km³/s²
        return 2 * np.pi * np.sqrt(self.semi_major_axis**3 / mu)
    
    @property
    def mean_motion(self) -> float:
        """Calculate mean motion in degrees per second."""
        return 360.0 / self.period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'semi_major_axis': self.semi_major_axis,
            'eccentricity': self.eccentricity,
            'inclination': self.inclination,
            'raan': self.raan,
            'arg_periapsis': self.arg_periapsis,
            'true_anomaly': self.true_anomaly,
            'epoch': self.epoch.isoformat(),
            'period': self.period,
            'mean_motion': self.mean_motion
        }


@dataclass
class StateVector:
    """Cartesian state vector (position and velocity)."""
    
    position: np.ndarray  # [x, y, z] in km
    velocity: np.ndarray  # [vx, vy, vz] in km/s
    timestamp: datetime
    
    def __post_init__(self):
        """Validate state vector."""
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3-element array")
        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be a 3-element array")
    
    @property
    def speed(self) -> float:
        """Calculate orbital speed in km/s."""
        return np.linalg.norm(self.velocity)
    
    @property
    def altitude(self) -> float:
        """Calculate altitude above Earth surface in km."""
        earth_radius = 6378.137  # km
        return np.linalg.norm(self.position) - earth_radius
    
    def distance_to(self, other: 'StateVector') -> float:
        """Calculate distance to another state vector in km."""
        return np.linalg.norm(self.position - other.position)
    
    def relative_velocity(self, other: 'StateVector') -> np.ndarray:
        """Calculate relative velocity vector to another state vector."""
        return self.velocity - other.velocity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'timestamp': self.timestamp.isoformat(),
            'speed': self.speed,
            'altitude': self.altitude
        }


class SatelliteState:
    """
    Complete satellite state including orbital elements, state vectors, and metadata.
    """
    
    def __init__(self, 
                 satellite_id: str,
                 elements: Optional[OrbitalElements] = None,
                 state_vector: Optional[StateVector] = None,
                 mass: float = 1000.0,  # kg
                 cross_sectional_area: float = 10.0,  # m²
                 drag_coefficient: float = 2.2,
                 reflectivity_coefficient: float = 1.3,
                 operational_status: str = "active"):
        """
        Initialize satellite state.
        
        Args:
            satellite_id: Unique identifier for the satellite
            elements: Keplerian orbital elements
            state_vector: Cartesian state vector
            mass: Satellite mass in kg
            cross_sectional_area: Cross-sectional area in m²
            drag_coefficient: Atmospheric drag coefficient
            reflectivity_coefficient: Solar radiation pressure coefficient
            operational_status: Current operational status
        """
        self.satellite_id = satellite_id
        self.elements = elements
        self.state_vector = state_vector
        self.mass = mass
        self.cross_sectional_area = cross_sectional_area
        self.drag_coefficient = drag_coefficient
        self.reflectivity_coefficient = reflectivity_coefficient
        self.operational_status = operational_status
        
        # State history for tracking
        self.state_history = []
        if state_vector:
            self.state_history.append(state_vector)
    
    def update_state_vector(self, new_state: StateVector):
        """Update the current state vector and add to history."""
        self.state_vector = new_state
        self.state_history.append(new_state)
        
        # Keep only last 24 hours of history (assuming 1 Hz updates)
        max_history = 24 * 3600
        if len(self.state_history) > max_history:
            self.state_history = self.state_history[-max_history:]
    
    def update_elements(self, new_elements: OrbitalElements):
        """Update the orbital elements."""
        self.elements = new_elements
    
    def get_state_at_time(self, timestamp: datetime) -> Optional[StateVector]:
        """Get state vector closest to specified time."""
        if not self.state_history:
            return None
        
        # Find closest state by timestamp
        min_diff = float('inf')
        closest_state = None
        
        for state in self.state_history:
            diff = abs((state.timestamp - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_state = state
        
        return closest_state
    
    @property
    def current_altitude(self) -> Optional[float]:
        """Get current altitude in km."""
        if self.state_vector:
            return self.state_vector.altitude
        return None
    
    @property
    def current_speed(self) -> Optional[float]:
        """Get current orbital speed in km/s."""
        if self.state_vector:
            return self.state_vector.speed
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'satellite_id': self.satellite_id,
            'elements': self.elements.to_dict() if self.elements else None,
            'state_vector': self.state_vector.to_dict() if self.state_vector else None,
            'mass': self.mass,
            'cross_sectional_area': self.cross_sectional_area,
            'drag_coefficient': self.drag_coefficient,
            'reflectivity_coefficient': self.reflectivity_coefficient,
            'operational_status': self.operational_status,
            'current_altitude': self.current_altitude,
            'current_speed': self.current_speed,
            'history_length': len(self.state_history)
        }
    
    def __repr__(self) -> str:
        """String representation of satellite state."""
        alt = f"{self.current_altitude:.2f} km" if self.current_altitude else "Unknown"
        speed = f"{self.current_speed:.3f} km/s" if self.current_speed else "Unknown"
        return f"SatelliteState(id={self.satellite_id}, alt={alt}, speed={speed}, status={self.operational_status})" 