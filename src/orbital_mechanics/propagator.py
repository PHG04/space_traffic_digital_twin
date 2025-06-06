"""
Orbital Propagation Algorithms

Implementation of Keplerian and J2 perturbed orbital propagation.
"""

import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .satellite_state import OrbitalElements, StateVector


class OrbitPropagator(ABC):
    """Abstract base class for orbit propagators."""
    
    @abstractmethod
    def propagate(self, elements: OrbitalElements, dt: float) -> OrbitalElements:
        """
        Propagate orbital elements forward in time.
        
        Args:
            elements: Initial orbital elements
            dt: Time step in seconds
            
        Returns:
            Propagated orbital elements
        """
        pass


class KeplerianPropagator(OrbitPropagator):
    """
    Pure Keplerian orbital propagation (no perturbations).
    """
    
    def __init__(self, mu: float = 398600.4418):
        """
        Initialize Keplerian propagator.
        
        Args:
            mu: Earth's gravitational parameter in km³/s²
        """
        self.mu = mu
    
    def propagate(self, elements: OrbitalElements, dt: float) -> OrbitalElements:
        """Propagate using Keplerian orbital mechanics."""
        # Calculate mean motion
        n = np.sqrt(self.mu / elements.semi_major_axis**3)  # rad/s
        
        # Calculate mean anomaly change
        M0 = self.true_to_mean_anomaly(elements.true_anomaly, elements.eccentricity)
        M = M0 + n * dt
        
        # Convert back to true anomaly
        new_true_anomaly = self.mean_to_true_anomaly(M, elements.eccentricity)
        
        # Create new elements (only true anomaly changes in pure Keplerian motion)
        new_epoch = elements.epoch + timedelta(seconds=dt)
        
        return OrbitalElements(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination,
            raan=elements.raan,
            arg_periapsis=elements.arg_periapsis,
            true_anomaly=np.degrees(new_true_anomaly),
            epoch=new_epoch
        )
    
    @staticmethod
    def true_to_mean_anomaly(true_anomaly_deg: float, eccentricity: float) -> float:
        """Convert true anomaly to mean anomaly (radians)."""
        nu = np.radians(true_anomaly_deg)
        
        # Eccentric anomaly
        E = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(nu / 2))
        
        # Mean anomaly
        M = E - eccentricity * np.sin(E)
        
        return M
    
    @staticmethod
    def mean_to_true_anomaly(mean_anomaly_rad: float, eccentricity: float, tolerance: float = 1e-12) -> float:
        """Convert mean anomaly to true anomaly using Newton-Raphson method."""
        M = mean_anomaly_rad
        e = eccentricity
        
        # Initial guess for eccentric anomaly
        E = M if e < 0.8 else np.pi
        
        # Newton-Raphson iteration to solve Kepler's equation
        for _ in range(50):  # Maximum iterations
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            
            if abs(f) < tolerance:
                break
                
            E = E - f / fp
        
        # Convert eccentric anomaly to true anomaly
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        
        return nu
    
    def elements_to_state_vector(self, elements: OrbitalElements) -> StateVector:
        """Convert orbital elements to Cartesian state vector."""
        # Convert to radians
        nu = np.radians(elements.true_anomaly)
        omega = np.radians(elements.arg_periapsis)
        Omega = np.radians(elements.raan)
        i = np.radians(elements.inclination)
        
        a = elements.semi_major_axis
        e = elements.eccentricity
        
        # Distance from focus
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        
        # Position and velocity in orbital plane
        r_orb = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        
        # Velocity magnitude
        v_mag = np.sqrt(self.mu * (2/r - 1/a))
        
        # Velocity direction (perpendicular to position in orbital plane)
        h = np.sqrt(self.mu * a * (1 - e**2))  # Specific angular momentum
        v_orb = np.array([
            -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu),
            np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu)),
            0
        ])
        
        # Transformation matrices
        R1 = np.array([
            [np.cos(Omega), -np.sin(Omega), 0],
            [np.sin(Omega), np.cos(Omega), 0],
            [0, 0, 1]
        ])
        
        R2 = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3 = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1]
        ])
        
        # Combined transformation from orbital to inertial frame
        R = R1 @ R2 @ R3
        
        # Transform to inertial frame
        r_inertial = R @ r_orb
        v_inertial = R @ v_orb
        
        return StateVector(
            position=r_inertial,
            velocity=v_inertial,
            timestamp=elements.epoch
        )


class J2Propagator(KeplerianPropagator):
    """
    Orbital propagation including J2 perturbation effects.
    """
    
    def __init__(self, mu: float = 398600.4418, j2: float = 1.08262668e-3, re: float = 6378.137):
        """
        Initialize J2 propagator.
        
        Args:
            mu: Earth's gravitational parameter in km³/s²
            j2: J2 coefficient (Earth's oblateness)
            re: Earth's equatorial radius in km
        """
        super().__init__(mu)
        self.j2 = j2
        self.re = re
    
    def propagate(self, elements: OrbitalElements, dt: float) -> OrbitalElements:
        """Propagate using Keplerian motion with J2 perturbations."""
        a = elements.semi_major_axis
        e = elements.eccentricity
        i = np.radians(elements.inclination)
        
        # Calculate J2 perturbation rates
        n = np.sqrt(self.mu / a**3)  # Mean motion
        p = a * (1 - e**2)  # Semi-latus rectum
        
        # J2 perturbation effects (secular rates)
        factor = -1.5 * self.j2 * (self.re**2) * n / (p**2)
        
        # Rate of change of RAAN (node regression)
        domega_dt = factor * np.cos(i)
        
        # Rate of change of argument of periapsis
        dw_dt = factor * (2 - 2.5 * np.sin(i)**2)
        
        # Rate of change of mean anomaly (modified by J2)
        dM_dt = n + factor * np.sqrt(1 - e**2) * (1 - 1.5 * np.sin(i)**2)
        
        # Apply perturbations
        new_raan = elements.raan + np.degrees(domega_dt * dt)
        new_arg_periapsis = elements.arg_periapsis + np.degrees(dw_dt * dt)
        
        # Handle mean anomaly evolution
        M0 = self.true_to_mean_anomaly(elements.true_anomaly, e)
        M = M0 + dM_dt * dt
        new_true_anomaly = np.degrees(self.mean_to_true_anomaly(M, e))
        
        # Normalize angles
        new_raan = new_raan % 360
        new_arg_periapsis = new_arg_periapsis % 360
        new_true_anomaly = new_true_anomaly % 360
        
        new_epoch = elements.epoch + timedelta(seconds=dt)
        
        return OrbitalElements(
            semi_major_axis=a,  # No change in semi-major axis for J2
            eccentricity=e,     # No change in eccentricity for J2
            inclination=elements.inclination,  # No change in inclination for J2
            raan=new_raan,
            arg_periapsis=new_arg_periapsis,
            true_anomaly=new_true_anomaly,
            epoch=new_epoch
        )


class StatePropagator:
    """
    High-level state propagator that manages both orbital elements and state vectors.
    """
    
    def __init__(self, use_j2: bool = True):
        """
        Initialize state propagator.
        
        Args:
            use_j2: Whether to include J2 perturbation effects
        """
        self.propagator = J2Propagator() if use_j2 else KeplerianPropagator()
        self.use_j2 = use_j2
    
    def propagate_state(self, satellite_state: 'SatelliteState', dt: float) -> 'StateVector':
        """
        Propagate satellite state forward in time.
        
        Args:
            satellite_state: Current satellite state
            dt: Time step in seconds
            
        Returns:
            New state vector at propagated time
        """
        if not satellite_state.elements:
            raise ValueError("Satellite state must have orbital elements for propagation")
        
        # Propagate orbital elements
        new_elements = self.propagator.propagate(satellite_state.elements, dt)
        
        # Convert to state vector
        new_state_vector = self.propagator.elements_to_state_vector(new_elements)
        
        return new_state_vector
    
    def batch_propagate(self, satellite_states: list, dt: float) -> dict:
        """
        Propagate multiple satellites in batch.
        
        Args:
            satellite_states: List of satellite states
            dt: Time step in seconds
            
        Returns:
            Dictionary mapping satellite IDs to new state vectors
        """
        results = {}
        
        for sat_state in satellite_states:
            try:
                new_state = self.propagate_state(sat_state, dt)
                results[sat_state.satellite_id] = new_state
            except Exception as e:
                print(f"Error propagating satellite {sat_state.satellite_id}: {e}")
                results[sat_state.satellite_id] = None
        
        return results 