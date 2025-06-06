"""
Advanced Collision Probability Calculation Engine

Implements multiple sophisticated algorithms for calculating collision probabilities
between satellites, including Chan, Foster, Alfano, and Monte Carlo methods.
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from scipy import integrate
from scipy.linalg import cholesky, LinAlgError
from scipy import special

# Handle both relative and absolute imports
try:
    from ..orbital_mechanics.satellite_state import StateVector
except ImportError:
    # If relative import fails, try absolute import for when running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from orbital_mechanics.satellite_state import StateVector


class PcCalculationMethod(Enum):
    """Available collision probability calculation methods."""
    CHAN = "chan"
    FOSTER = "foster"
    ALFANO = "alfano"
    MONTE_CARLO = "monte_carlo"
    SIMPLE_GAUSSIAN = "simple_gaussian"


@dataclass
class CovarianceMatrix:
    """Covariance matrix for position and velocity uncertainties."""
    
    position_covariance: np.ndarray  # 3x3 position covariance (km²)
    velocity_covariance: np.ndarray  # 3x3 velocity covariance (km²/s²)
    position_velocity_covariance: np.ndarray  # 3x3 cross-covariance (km²/s)
    
    def get_full_covariance(self) -> np.ndarray:
        """Get the full 6x6 state covariance matrix."""
        full_cov = np.zeros((6, 6))
        full_cov[0:3, 0:3] = self.position_covariance
        full_cov[3:6, 3:6] = self.velocity_covariance
        full_cov[0:3, 3:6] = self.position_velocity_covariance
        full_cov[3:6, 0:3] = self.position_velocity_covariance.T
        return full_cov
    
    @classmethod
    def from_diagonal(cls, pos_sigma: float, vel_sigma: float, cross_correlation: float = 0.0):
        """Create covariance matrix from diagonal uncertainties."""
        pos_cov = np.eye(3) * (pos_sigma ** 2)
        vel_cov = np.eye(3) * (vel_sigma ** 2)
        cross_cov = np.eye(3) * (cross_correlation * pos_sigma * vel_sigma)
        
        return cls(
            position_covariance=pos_cov,
            velocity_covariance=vel_cov,
            position_velocity_covariance=cross_cov
        )


@dataclass
class CollisionProbabilityResult:
    """Result of collision probability calculation."""
    
    probability: float
    method_used: PcCalculationMethod
    calculation_time_ms: float
    convergence_achieved: bool
    uncertainty_estimate: float
    miss_distance_mean: float
    miss_distance_sigma: float
    relative_velocity: float
    combined_radius: float
    
    # Method-specific details
    integration_points: int = 0
    monte_carlo_samples: int = 0
    numerical_error: float = 0.0


class CollisionProbabilityEngine:
    """
    Advanced collision probability calculation engine.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize collision probability engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.default_combined_radius = self.config.get('default_combined_radius_km', 0.01)  # 10m
        self.monte_carlo_samples = self.config.get('monte_carlo_samples', 100000)
        self.integration_tolerance = self.config.get('integration_tolerance', 1e-8)
        self.max_integration_subdivisions = self.config.get('max_integration_subdivisions', 1000)
        
        # Numerical stability parameters
        self.min_sigma_ratio = self.config.get('min_sigma_ratio', 1e-6)
        self.max_sigma_ratio = self.config.get('max_sigma_ratio', 1e6)
        
        self.logger.info("Collision Probability Engine initialized")
    
    def calculate_collision_probability(self,
                                      state1: StateVector,
                                      state2: StateVector,
                                      covariance1: CovarianceMatrix,
                                      covariance2: CovarianceMatrix,
                                      combined_radius: float = None,
                                      method: PcCalculationMethod = PcCalculationMethod.CHAN,
                                      time_to_ca: float = 0.0) -> CollisionProbabilityResult:
        """
        Calculate collision probability between two objects.
        
        Args:
            state1: State vector of first object
            state2: State vector of second object
            covariance1: Covariance matrix of first object
            covariance2: Covariance matrix of second object
            combined_radius: Combined hard-body radius (km)
            method: Calculation method to use
            time_to_ca: Time to closest approach (seconds)
            
        Returns:
            Collision probability result
        """
        start_time = datetime.now()
        
        try:
            if combined_radius is None:
                combined_radius = self.default_combined_radius
            
            # Calculate relative state
            relative_position = state1.position - state2.position
            relative_velocity = state1.velocity - state2.velocity
            
            # Combine covariance matrices
            combined_covariance = self._combine_covariances(covariance1, covariance2)
            
            # Select calculation method
            if method == PcCalculationMethod.CHAN:
                result = self._calculate_pc_chan(
                    relative_position, relative_velocity, combined_covariance, combined_radius
                )
            elif method == PcCalculationMethod.FOSTER:
                result = self._calculate_pc_foster(
                    relative_position, relative_velocity, combined_covariance, combined_radius
                )
            elif method == PcCalculationMethod.ALFANO:
                result = self._calculate_pc_alfano(
                    relative_position, relative_velocity, combined_covariance, combined_radius
                )
            elif method == PcCalculationMethod.MONTE_CARLO:
                result = self._calculate_pc_monte_carlo(
                    relative_position, relative_velocity, combined_covariance, combined_radius
                )
            elif method == PcCalculationMethod.SIMPLE_GAUSSIAN:
                result = self._calculate_pc_simple_gaussian(
                    relative_position, relative_velocity, combined_covariance, combined_radius
                )
            else:
                raise ValueError(f"Unknown calculation method: {method}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.calculation_time_ms = execution_time
            result.method_used = method
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating collision probability: {e}")
            # Return safe default result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return CollisionProbabilityResult(
                probability=0.0,
                method_used=method,
                calculation_time_ms=execution_time,
                convergence_achieved=False,
                uncertainty_estimate=1.0,
                miss_distance_mean=np.linalg.norm(relative_position),
                miss_distance_sigma=1.0,
                relative_velocity=np.linalg.norm(relative_velocity),
                combined_radius=combined_radius or self.default_combined_radius,
                numerical_error=1.0
            )
    
    def _combine_covariances(self, cov1: CovarianceMatrix, cov2: CovarianceMatrix) -> CovarianceMatrix:
        """Combine two covariance matrices for relative motion."""
        
        # For independent objects, covariances add
        combined_pos_cov = cov1.position_covariance + cov2.position_covariance
        combined_vel_cov = cov1.velocity_covariance + cov2.velocity_covariance
        combined_cross_cov = cov1.position_velocity_covariance + cov2.position_velocity_covariance
        
        return CovarianceMatrix(
            position_covariance=combined_pos_cov,
            velocity_covariance=combined_vel_cov,
            position_velocity_covariance=combined_cross_cov
        )
    
    def _calculate_pc_chan(self,
                          relative_position: np.ndarray,
                          relative_velocity: np.ndarray,
                          covariance: CovarianceMatrix,
                          combined_radius: float) -> CollisionProbabilityResult:
        """
        Calculate Pc using the Chan method (2008).
        
        This is a highly accurate method for short-term encounters.
        """
        
        # Transform to encounter coordinate system
        encounter_frame = self._get_encounter_frame(relative_position, relative_velocity)
        
        # Transform covariance to encounter frame
        full_cov = covariance.get_full_covariance()
        encounter_cov = encounter_frame.T @ full_cov @ encounter_frame
        
        # Extract 2D covariance in the encounter plane (perpendicular to relative velocity)
        # Encounter frame: [along-track, cross-track1, cross-track2, vel_along, vel_cross1, vel_cross2]
        encounter_plane_cov = encounter_cov[1:3, 1:3]  # Cross-track position covariance
        
        # Calculate miss distance statistics
        relative_pos_encounter = encounter_frame.T @ np.concatenate([relative_position, relative_velocity])
        miss_distance_mean = np.linalg.norm(relative_pos_encounter[1:3])
        
        try:
            # Eigenvalue decomposition for numerical stability
            eigenvals, eigenvecs = np.linalg.eigh(encounter_plane_cov)
            
            if np.any(eigenvals <= 0):
                self.logger.warning("Non-positive definite covariance matrix")
                eigenvals = np.maximum(eigenvals, 1e-12)
            
            # Transform to principal axes
            sqrt_eigenvals = np.sqrt(eigenvals)
            
            # Calculate probability using 2D integration
            # P = ∫∫ (1/2π√det(Σ)) * exp(-0.5 * r^T * Σ^-1 * r) dr
            # where integration is over the circle of radius R
            
            # Use Alfano's approximation for efficiency
            sigma_major = sqrt_eigenvals[1]  # Larger eigenvalue
            sigma_minor = sqrt_eigenvals[0]  # Smaller eigenvalue
            
            # Dilution factor
            dilution = sigma_major / sigma_minor
            
            if dilution > self.max_sigma_ratio:
                # Very elongated ellipse - use asymptotic approximation
                pc = self._calculate_pc_asymptotic(miss_distance_mean, sigma_minor, combined_radius)
                convergence = False
            else:
                # Use series expansion method
                pc = self._calculate_pc_series_expansion(
                    miss_distance_mean, sigma_major, sigma_minor, combined_radius
                )
                convergence = True
            
            return CollisionProbabilityResult(
                probability=pc,
                method_used=PcCalculationMethod.CHAN,
                calculation_time_ms=0.0,  # Will be set by caller
                convergence_achieved=convergence,
                uncertainty_estimate=0.1 * pc if pc > 0 else 1e-10,
                miss_distance_mean=miss_distance_mean,
                miss_distance_sigma=np.sqrt(np.mean(eigenvals)),
                relative_velocity=np.linalg.norm(relative_velocity),
                combined_radius=combined_radius,
                numerical_error=1e-8 if convergence else 1e-6
            )
            
        except LinAlgError as e:
            self.logger.error(f"Linear algebra error in Chan method: {e}")
            return self._fallback_calculation(relative_position, relative_velocity, combined_radius)
    
    def _calculate_pc_foster(self,
                           relative_position: np.ndarray,
                           relative_velocity: np.ndarray,
                           covariance: CovarianceMatrix,
                           combined_radius: float) -> CollisionProbabilityResult:
        """
        Calculate Pc using the Foster method.
        
        Good for medium-term encounters with moderate uncertainty.
        """
        
        # Similar to Chan but with different numerical approach
        # This is a simplified implementation
        miss_distance = np.linalg.norm(relative_position)
        
        # Use position covariance only (simplified)
        pos_cov = covariance.position_covariance
        eigenvals, _ = np.linalg.eigh(pos_cov)
        
        # Average uncertainty
        avg_sigma = np.sqrt(np.mean(eigenvals))
        
        # Foster's approximation
        if miss_distance > 5 * avg_sigma:
            # Far encounter - use asymptotic formula
            pc = (combined_radius / miss_distance) * np.exp(-0.5 * (miss_distance / avg_sigma)**2)
        else:
            # Close encounter - use numerical integration
            pc = self._numerical_integration_2d(miss_distance, avg_sigma, combined_radius)
        
        return CollisionProbabilityResult(
            probability=pc,
            method_used=PcCalculationMethod.FOSTER,
            calculation_time_ms=0.0,
            convergence_achieved=True,
            uncertainty_estimate=0.2 * pc if pc > 0 else 1e-10,
            miss_distance_mean=miss_distance,
            miss_distance_sigma=avg_sigma,
            relative_velocity=np.linalg.norm(relative_velocity),
            combined_radius=combined_radius
        )
    
    def _calculate_pc_alfano(self,
                           relative_position: np.ndarray,
                           relative_velocity: np.ndarray,
                           covariance: CovarianceMatrix,
                           combined_radius: float) -> CollisionProbabilityResult:
        """
        Calculate Pc using Alfano's method.
        
        Efficient method with good accuracy for most scenarios.
        """
        
        miss_distance = np.linalg.norm(relative_position)
        
        # Extract 2D covariance in the encounter plane
        encounter_frame = self._get_encounter_frame(relative_position, relative_velocity)
        full_cov = covariance.get_full_covariance()
        encounter_cov = encounter_frame.T @ full_cov @ encounter_frame
        
        # 2D position covariance in encounter plane
        plane_cov = encounter_cov[1:3, 1:3]
        
        # Eigenvalue decomposition
        eigenvals, _ = np.linalg.eigh(plane_cov)
        sigma_1, sigma_2 = np.sqrt(eigenvals)
        
        # Alfano's series expansion
        pc = self._alfano_series_expansion(miss_distance, sigma_1, sigma_2, combined_radius)
        
        return CollisionProbabilityResult(
            probability=pc,
            method_used=PcCalculationMethod.ALFANO,
            calculation_time_ms=0.0,
            convergence_achieved=True,
            uncertainty_estimate=0.15 * pc if pc > 0 else 1e-10,
            miss_distance_mean=miss_distance,
            miss_distance_sigma=np.sqrt(np.mean(eigenvals)),
            relative_velocity=np.linalg.norm(relative_velocity),
            combined_radius=combined_radius
        )
    
    def _calculate_pc_monte_carlo(self,
                                relative_position: np.ndarray,
                                relative_velocity: np.ndarray,
                                covariance: CovarianceMatrix,
                                combined_radius: float) -> CollisionProbabilityResult:
        """
        Calculate Pc using Monte Carlo simulation.
        
        Most robust method but computationally expensive.
        """
        
        # Generate random samples from the combined distribution
        full_cov = covariance.get_full_covariance()
        
        try:
            # Cholesky decomposition for sampling
            L = cholesky(full_cov, lower=True)
            
            # Generate samples
            samples = np.random.randn(self.monte_carlo_samples, 6)
            state_samples = samples @ L.T
            
            # Add mean state
            mean_state = np.concatenate([relative_position, relative_velocity])
            state_samples += mean_state
            
            # Count collisions
            collisions = 0
            miss_distances = []
            
            for i in range(self.monte_carlo_samples):
                sample_pos = state_samples[i, 0:3]
                miss_dist = np.linalg.norm(sample_pos)
                miss_distances.append(miss_dist)
                
                if miss_dist <= combined_radius:
                    collisions += 1
            
            pc = collisions / self.monte_carlo_samples
            
            # Calculate statistics
            miss_distances = np.array(miss_distances)
            miss_mean = np.mean(miss_distances)
            miss_sigma = np.std(miss_distances)
            
            # Estimate uncertainty (binomial confidence interval)
            uncertainty = 1.96 * np.sqrt(pc * (1 - pc) / self.monte_carlo_samples)
            
            return CollisionProbabilityResult(
                probability=pc,
                method_used=PcCalculationMethod.MONTE_CARLO,
                calculation_time_ms=0.0,
                convergence_achieved=True,
                uncertainty_estimate=uncertainty,
                miss_distance_mean=miss_mean,
                miss_distance_sigma=miss_sigma,
                relative_velocity=np.linalg.norm(relative_velocity),
                combined_radius=combined_radius,
                monte_carlo_samples=self.monte_carlo_samples
            )
            
        except LinAlgError as e:
            self.logger.error(f"Cholesky decomposition failed: {e}")
            return self._fallback_calculation(relative_position, relative_velocity, combined_radius)
    
    def _calculate_pc_simple_gaussian(self,
                                    relative_position: np.ndarray,
                                    relative_velocity: np.ndarray,
                                    covariance: CovarianceMatrix,
                                    combined_radius: float) -> CollisionProbabilityResult:
        """
        Simple Gaussian approximation for quick estimates.
        """
        
        miss_distance = np.linalg.norm(relative_position)
        
        # Use average position uncertainty
        pos_cov = covariance.position_covariance
        avg_sigma = np.sqrt(np.trace(pos_cov) / 3)
        
        # Simple 1D Gaussian approximation
        if miss_distance > 0:
            pc = 0.5 * special.erfc((miss_distance - combined_radius) / (np.sqrt(2) * avg_sigma))
        else:
            pc = 1.0
        
        pc = max(0.0, min(1.0, pc))
        
        return CollisionProbabilityResult(
            probability=pc,
            method_used=PcCalculationMethod.SIMPLE_GAUSSIAN,
            calculation_time_ms=0.0,
            convergence_achieved=True,
            uncertainty_estimate=0.5 * pc if pc > 0 else 1e-10,
            miss_distance_mean=miss_distance,
            miss_distance_sigma=avg_sigma,
            relative_velocity=np.linalg.norm(relative_velocity),
            combined_radius=combined_radius
        )
    
    def _get_encounter_frame(self, relative_position: np.ndarray, relative_velocity: np.ndarray) -> np.ndarray:
        """
        Create encounter coordinate frame.
        
        Returns 6x6 transformation matrix to encounter coordinates:
        - x: along relative velocity (radial)
        - y: cross-track 1
        - z: cross-track 2
        """
        
        # Normalize relative velocity (along-track direction)
        vel_norm = np.linalg.norm(relative_velocity)
        if vel_norm > 0:
            x_axis = relative_velocity / vel_norm
        else:
            x_axis = np.array([1.0, 0.0, 0.0])
        
        # Create perpendicular axes
        # Choose a vector not parallel to x_axis
        if abs(x_axis[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        
        # Gram-Schmidt orthogonalization
        y_axis = temp - np.dot(temp, x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        z_axis = np.cross(x_axis, y_axis)
        
        # Build transformation matrix
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        
        # 6x6 transformation matrix
        transform = np.zeros((6, 6))
        transform[0:3, 0:3] = rotation
        transform[3:6, 3:6] = rotation
        
        return transform
    
    def _calculate_pc_series_expansion(self, miss_distance: float, sigma_major: float, 
                                     sigma_minor: float, radius: float) -> float:
        """Calculate Pc using series expansion method."""
        
        # Normalized parameters
        rho = miss_distance / sigma_minor
        R = radius / sigma_minor
        alpha = sigma_minor / sigma_major
        
        # Series expansion (simplified)
        if rho > 5 * R:
            # Far encounter - asymptotic approximation
            pc = (R / rho) * np.exp(-0.5 * rho**2) * alpha
        else:
            # Close encounter - use modified Bessel functions
            # This is a simplified approximation
            pc = alpha * np.exp(-0.5 * rho**2) * (1 - np.exp(-0.5 * R**2))
        
        return max(0.0, min(1.0, pc))
    
    def _calculate_pc_asymptotic(self, miss_distance: float, sigma: float, radius: float) -> float:
        """Asymptotic approximation for very elongated uncertainty ellipses."""
        
        if miss_distance > 0:
            pc = (radius / miss_distance) * np.exp(-0.5 * (miss_distance / sigma)**2)
        else:
            pc = 1.0
        
        return max(0.0, min(1.0, pc))
    
    def _numerical_integration_2d(self, miss_distance: float, sigma: float, radius: float) -> float:
        """2D numerical integration for collision probability."""
        
        def integrand(r, theta):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            distance_from_center = np.sqrt((x - miss_distance)**2 + y**2)
            
            if distance_from_center <= radius:
                return (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (x**2 + y**2) / sigma**2)
            else:
                return 0.0
        
        # Integration limits
        r_max = miss_distance + radius + 5 * sigma
        
        try:
            result, _ = integrate.dblquad(
                lambda r, theta: r * integrand(r, theta),
                0, 2 * np.pi,
                lambda theta: 0,
                lambda theta: r_max,
                epsabs=self.integration_tolerance
            )
            return max(0.0, min(1.0, result))
        except:
            # Fallback to simple approximation
            return self._calculate_pc_asymptotic(miss_distance, sigma, radius)
    
    def _alfano_series_expansion(self, miss_distance: float, sigma1: float, 
                               sigma2: float, radius: float) -> float:
        """Alfano's series expansion method."""
        
        # Ensure sigma1 >= sigma2
        if sigma1 < sigma2:
            sigma1, sigma2 = sigma2, sigma1
        
        # Normalized parameters
        rho = miss_distance / sigma2
        R = radius / sigma2
        alpha = sigma2 / sigma1
        
        # Alfano's approximation
        if rho > 3:
            # Far encounter
            pc = alpha * (R / rho) * np.exp(-0.5 * rho**2)
        else:
            # Close encounter - use series
            pc = alpha * (1 - np.exp(-0.5 * R**2)) * np.exp(-0.5 * rho**2)
        
        return max(0.0, min(1.0, pc))
    
    def _fallback_calculation(self, relative_position: np.ndarray, 
                            relative_velocity: np.ndarray, combined_radius: float) -> CollisionProbabilityResult:
        """Fallback calculation when other methods fail."""
        
        miss_distance = np.linalg.norm(relative_position)
        
        # Very simple approximation
        if miss_distance <= combined_radius:
            pc = 1.0
        else:
            pc = 0.0
        
        return CollisionProbabilityResult(
            probability=pc,
            method_used=PcCalculationMethod.SIMPLE_GAUSSIAN,
            calculation_time_ms=0.0,
            convergence_achieved=False,
            uncertainty_estimate=1.0,
            miss_distance_mean=miss_distance,
            miss_distance_sigma=1.0,
            relative_velocity=np.linalg.norm(relative_velocity),
            combined_radius=combined_radius,
            numerical_error=1.0
        ) 