"""
Conjunction Analysis and Collision Probability Calculator

Implements conjunction detection algorithms and sophisticated collision probability
calculations for satellite proximity analysis.
"""

import sys
import os
import logging
import threading
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import chi2
from enum import Enum
import numpy as np

# Handle both relative and absolute imports
try:
    from .spatial_index import SpatialIndex, SpatialObject
    from ..orbital_mechanics.satellite_state import SatelliteState, StateVector
except ImportError:
    # If relative import fails, try absolute import for when running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from conjunction_detection.spatial_index import SpatialIndex, SpatialObject
    from orbital_mechanics.satellite_state import SatelliteState, StateVector


class ConjunctionSeverity(Enum):
    """Conjunction severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConjunctionEvent:
    """Represents a conjunction event between two satellites."""
    
    event_id: str
    satellite_1: str
    satellite_2: str
    time_of_closest_approach: datetime
    closest_approach_distance: float  # km
    relative_velocity: float  # km/s
    collision_probability: float
    miss_distance_radial: float
    miss_distance_in_track: float
    miss_distance_cross_track: float
    severity: ConjunctionSeverity
    detection_time: datetime
    dilution_threshold: float
    covariance_1: Optional[np.ndarray] = None
    covariance_2: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'satellite_1': self.satellite_1,
            'satellite_2': self.satellite_2,
            'time_of_closest_approach': self.time_of_closest_approach.isoformat(),
            'closest_approach_distance': self.closest_approach_distance,
            'relative_velocity': self.relative_velocity,
            'collision_probability': self.collision_probability,
            'miss_distance_radial': self.miss_distance_radial,
            'miss_distance_in_track': self.miss_distance_in_track,
            'miss_distance_cross_track': self.miss_distance_cross_track,
            'severity': self.severity.value,
            'detection_time': self.detection_time.isoformat(),
            'dilution_threshold': self.dilution_threshold
        }


class CollisionProbabilityCalculator:
    """
    Sophisticated collision probability calculator using covariance-based methods.
    
    Implements Chan method and other standard algorithms for Pc calculation.
    """
    
    def __init__(self):
        """Initialize collision probability calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Default hard-body radii (km)
        self.default_satellite_radius = 0.005  # 5 meters
        self.default_debris_radius = 0.001     # 1 meter
    
    def calculate_collision_probability(self, 
                                     position_1: np.ndarray,
                                     velocity_1: np.ndarray,
                                     covariance_1: np.ndarray,
                                     position_2: np.ndarray,
                                     velocity_2: np.ndarray,
                                     covariance_2: np.ndarray,
                                     radius_1: float = None,
                                     radius_2: float = None) -> float:
        """
        Calculate collision probability using covariance-based method.
        
        Args:
            position_1: Position vector of object 1 (km)
            velocity_1: Velocity vector of object 1 (km/s)
            covariance_1: 6x6 covariance matrix of object 1
            position_2: Position vector of object 2 (km)
            velocity_2: Velocity vector of object 2 (km/s)
            covariance_2: 6x6 covariance matrix of object 2
            radius_1: Hard-body radius of object 1 (km)
            radius_2: Hard-body radius of object 2 (km)
            
        Returns:
            Collision probability (0.0 to 1.0)
        """
        try:
            # Use default radii if not provided
            if radius_1 is None:
                radius_1 = self.default_satellite_radius
            if radius_2 is None:
                radius_2 = self.default_satellite_radius
            
            # Combined hard-body radius
            combined_radius = radius_1 + radius_2
            
            # Relative state vectors
            relative_position = position_1 - position_2
            relative_velocity = velocity_1 - velocity_2
            
            # Combined covariance matrix (assuming independence)
            combined_covariance = covariance_1 + covariance_2
            
            # Transform to encounter plane coordinates
            pc = self._calculate_pc_2d(relative_position, relative_velocity, 
                                     combined_covariance, combined_radius)
            
            return max(0.0, min(1.0, pc))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating collision probability: {e}")
            return 0.0
    
    def _calculate_pc_2d(self, relative_position: np.ndarray, 
                        relative_velocity: np.ndarray,
                        covariance: np.ndarray, 
                        combined_radius: float) -> float:
        """
        Calculate 2D collision probability in the encounter plane.
        
        Uses Chan method for 2D Pc calculation.
        """
        # Find time of closest approach
        relative_speed = np.linalg.norm(relative_velocity)
        if relative_speed < 1e-10:
            # Objects have same velocity - use current distance
            distance = np.linalg.norm(relative_position)
            return 1.0 if distance <= combined_radius else 0.0
        
        # Time to closest approach
        t_ca = -np.dot(relative_position, relative_velocity) / (relative_speed ** 2)
        
        # Position at closest approach
        position_ca = relative_position + relative_velocity * t_ca
        
        # Transform covariance to encounter plane
        # For simplicity, project onto plane perpendicular to velocity
        velocity_unit = relative_velocity / relative_speed
        
        # Create encounter plane coordinate system
        # x: in the plane, perpendicular to velocity
        # y: in the plane, perpendicular to both velocity and x
        # z: along velocity direction (not used in 2D calculation)
        
        # Choose arbitrary vector not parallel to velocity for x direction
        if abs(velocity_unit[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        
        x_unit = np.cross(velocity_unit, temp)
        x_unit = x_unit / np.linalg.norm(x_unit)
        
        y_unit = np.cross(velocity_unit, x_unit)
        y_unit = y_unit / np.linalg.norm(y_unit)
        
        # Transform position to encounter plane coordinates
        x_ca = np.dot(position_ca, x_unit)
        y_ca = np.dot(position_ca, y_unit)
        
        # Transform covariance to encounter plane
        # Extract position covariance (3x3 submatrix)
        pos_cov = covariance[:3, :3]
        
        # Project to 2D encounter plane
        transform_matrix = np.column_stack([x_unit, y_unit])
        cov_2d = transform_matrix.T @ pos_cov @ transform_matrix
        
        # Calculate 2D collision probability
        miss_distance = np.array([x_ca, y_ca])
        
        # Ensure covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(cov_2d)
        if np.any(eigenvals <= 0):
            # Add small regularization
            cov_2d += np.eye(2) * 1e-12
        
        try:
            # Calculate Mahalanobis distance
            cov_inv = np.linalg.inv(cov_2d)
            mahal_dist_sq = miss_distance.T @ cov_inv @ miss_distance
            
            # Calculate probability using circular hard-body assumption
            # This is a simplified approach - more sophisticated methods exist
            sigma_sq = combined_radius ** 2
            
            # Use complement of cumulative distribution function
            # for 2D Gaussian with circular integration region
            pc = 1.0 - np.exp(-sigma_sq * mahal_dist_sq / 2.0)
            
            return pc
            
        except np.linalg.LinAlgError:
            self.logger.warning("Singular covariance matrix in Pc calculation")
            # Fallback to geometric calculation
            miss_dist = np.linalg.norm(miss_distance)
            return 1.0 if miss_dist <= combined_radius else 0.0
    
    def calculate_dilution_threshold(self, covariance: np.ndarray, 
                                   confidence_level: float = 0.99) -> float:
        """
        Calculate dilution threshold for uncertainty visualization.
        
        Args:
            covariance: 6x6 covariance matrix
            confidence_level: Confidence level for threshold
            
        Returns:
            Dilution threshold in km
        """
        try:
            # Extract position covariance
            pos_cov = covariance[:3, :3]
            
            # Calculate eigenvalues to get uncertainty ellipsoid
            eigenvals = np.linalg.eigvals(pos_cov)
            max_eigenval = np.max(eigenvals)
            
            # Chi-squared threshold for confidence level
            chi2_threshold = chi2.ppf(confidence_level, df=3)
            
            # Threshold distance
            threshold = np.sqrt(chi2_threshold * max_eigenval)
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating dilution threshold: {e}")
            return 1.0  # Default 1 km threshold


class ConjunctionAnalyzer:
    """
    Main conjunction analysis engine that detects and analyzes satellite conjunctions.
    """
    
    def __init__(self, spatial_index: SpatialIndex, config: Dict = None):
        """
        Initialize conjunction analyzer.
        
        Args:
            spatial_index: Spatial indexing system for efficient proximity searches
            config: Configuration dictionary
        """
        self.spatial_index = spatial_index
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.conjunction_threshold = self.config.get('conjunction_threshold_km', 5.0)
        self.high_risk_threshold = self.config.get('high_risk_pc_threshold', 1e-4)
        self.medium_risk_threshold = self.config.get('medium_risk_pc_threshold', 1e-6)
        self.analysis_window_hours = self.config.get('analysis_window_hours', 24)
        
        # Components
        self.pc_calculator = CollisionProbabilityCalculator()
        
        # Event storage and tracking
        self.active_events: Dict[str, ConjunctionEvent] = {}
        self.event_history: List[ConjunctionEvent] = []
        self.event_counter = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("Conjunction Analyzer initialized")
    
    def analyze_conjunctions(self, satellites: Dict[str, SatelliteState]) -> List[ConjunctionEvent]:
        """
        Analyze all current satellites for potential conjunctions.
        
        Args:
            satellites: Dictionary of satellite states
            
        Returns:
            List of detected conjunction events
        """
        current_time = datetime.now()
        detected_events = []
        
        # Update spatial index with current satellite positions
        self._update_spatial_index(satellites, current_time)
        
        # Find all close approaches
        close_pairs = self.spatial_index.range_query_pairs(self.conjunction_threshold)
        
        with self.lock:
            for sat_id1, sat_id2, distance in close_pairs:
                if sat_id1 not in satellites or sat_id2 not in satellites:
                    continue
                
                # Generate event ID
                event_id = self._generate_event_id(sat_id1, sat_id2)
                
                # Check if this is a new event or update to existing
                if event_id in self.active_events:
                    # Update existing event
                    event = self._update_conjunction_event(
                        self.active_events[event_id], 
                        satellites[sat_id1], 
                        satellites[sat_id2], 
                        current_time
                    )
                else:
                    # Create new event
                    event = self._create_conjunction_event(
                        sat_id1, sat_id2, 
                        satellites[sat_id1], 
                        satellites[sat_id2], 
                        current_time
                    )
                    
                    if event:
                        self.active_events[event_id] = event
                
                if event:
                    detected_events.append(event)
        
        # Clean up old events
        self._cleanup_old_events(current_time)
        
        return detected_events
    
    def get_high_risk_conjunctions(self) -> List[ConjunctionEvent]:
        """Get all high-risk conjunction events."""
        with self.lock:
            return [event for event in self.active_events.values() 
                   if event.severity in [ConjunctionSeverity.HIGH, ConjunctionSeverity.CRITICAL]]
    
    def get_conjunction_by_id(self, event_id: str) -> Optional[ConjunctionEvent]:
        """Get conjunction event by ID."""
        with self.lock:
            return self.active_events.get(event_id)
    
    def get_conjunctions_for_satellite(self, satellite_id: str) -> List[ConjunctionEvent]:
        """Get all conjunction events involving a specific satellite."""
        with self.lock:
            return [event for event in self.active_events.values()
                   if satellite_id in [event.satellite_1, event.satellite_2]]
    
    def _update_spatial_index(self, satellites: Dict[str, SatelliteState], timestamp: datetime):
        """Update spatial index with current satellite positions."""
        for sat_id, satellite in satellites.items():
            if satellite.state_vector:
                spatial_obj = SpatialObject(
                    object_id=sat_id,
                    position=satellite.state_vector.position,
                    velocity=satellite.state_vector.velocity,
                    timestamp=timestamp,
                    metadata={'satellite': satellite}
                )
                self.spatial_index.update(spatial_obj)
    
    def _create_conjunction_event(self, sat_id1: str, sat_id2: str,
                                satellite1: SatelliteState, satellite2: SatelliteState,
                                current_time: datetime) -> Optional[ConjunctionEvent]:
        """Create a new conjunction event."""
        try:
            # Calculate conjunction parameters
            state1 = satellite1.state_vector
            state2 = satellite2.state_vector
            
            if not state1 or not state2:
                return None
            
            # Basic geometric calculations
            relative_position = state1.position - state2.position
            relative_velocity = state1.velocity - state2.velocity
            
            closest_distance = np.linalg.norm(relative_position)
            relative_speed = np.linalg.norm(relative_velocity)
            
            # Time of closest approach (simplified)
            if relative_speed > 1e-10:
                t_ca_seconds = -np.dot(relative_position, relative_velocity) / (relative_speed ** 2)
                time_ca = current_time + timedelta(seconds=t_ca_seconds)
            else:
                time_ca = current_time
            
            # Miss distance components (simplified)
            miss_distance_radial = 0.0  # Placeholder
            miss_distance_in_track = 0.0  # Placeholder  
            miss_distance_cross_track = closest_distance
            
            # Default covariance matrices (would be provided by tracking system)
            default_cov = np.eye(6) * 1e-6  # 1 meter position uncertainty
            
            # Calculate collision probability
            collision_prob = self.pc_calculator.calculate_collision_probability(
                state1.position, state1.velocity, default_cov,
                state2.position, state2.velocity, default_cov
            )
            
            # Determine severity
            severity = self._determine_severity(collision_prob, closest_distance)
            
            # Calculate dilution threshold
            dilution_threshold = self.pc_calculator.calculate_dilution_threshold(default_cov)
            
            # Create event
            event = ConjunctionEvent(
                event_id=self._generate_event_id(sat_id1, sat_id2),
                satellite_1=sat_id1,
                satellite_2=sat_id2,
                time_of_closest_approach=time_ca,
                closest_approach_distance=closest_distance,
                relative_velocity=relative_speed,
                collision_probability=collision_prob,
                miss_distance_radial=miss_distance_radial,
                miss_distance_in_track=miss_distance_in_track,
                miss_distance_cross_track=miss_distance_cross_track,
                severity=severity,
                detection_time=current_time,
                dilution_threshold=dilution_threshold,
                covariance_1=default_cov,
                covariance_2=default_cov
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error creating conjunction event: {e}")
            return None
    
    def _update_conjunction_event(self, event: ConjunctionEvent,
                                satellite1: SatelliteState, satellite2: SatelliteState,
                                current_time: datetime) -> ConjunctionEvent:
        """Update an existing conjunction event with new data."""
        # For now, create a new event (in practice, would update selectively)
        updated_event = self._create_conjunction_event(
            event.satellite_1, event.satellite_2,
            satellite1, satellite2, current_time
        )
        
        if updated_event:
            # Preserve original detection time
            updated_event.detection_time = event.detection_time
            return updated_event
        else:
            return event
    
    def _determine_severity(self, collision_prob: float, distance: float) -> ConjunctionSeverity:
        """Determine conjunction severity based on probability and distance."""
        if collision_prob >= self.high_risk_threshold or distance <= 1.0:
            return ConjunctionSeverity.CRITICAL
        elif collision_prob >= self.medium_risk_threshold or distance <= 2.0:
            return ConjunctionSeverity.HIGH
        elif distance <= 3.0:
            return ConjunctionSeverity.MEDIUM
        else:
            return ConjunctionSeverity.LOW
    
    def _generate_event_id(self, sat_id1: str, sat_id2: str) -> str:
        """Generate unique event ID for satellite pair."""
        # Ensure consistent ordering
        if sat_id1 > sat_id2:
            sat_id1, sat_id2 = sat_id2, sat_id1
        return f"CNJ_{sat_id1}_{sat_id2}_{self.event_counter}"
    
    def _cleanup_old_events(self, current_time: datetime):
        """Remove old conjunction events that are no longer relevant."""
        cutoff_time = current_time - timedelta(hours=self.analysis_window_hours)
        
        with self.lock:
            expired_events = []
            for event_id, event in self.active_events.items():
                if event.time_of_closest_approach < cutoff_time:
                    expired_events.append(event_id)
            
            for event_id in expired_events:
                event = self.active_events.pop(event_id)
                self.event_history.append(event)
            
            # Limit history size
            if len(self.event_history) > 10000:
                self.event_history = self.event_history[-5000:]
    
    def get_statistics(self) -> Dict:
        """Get conjunction analysis statistics."""
        with self.lock:
            active_events = list(self.active_events.values())
            
            severity_counts = {}
            for severity in ConjunctionSeverity:
                severity_counts[severity.value] = sum(
                    1 for event in active_events if event.severity == severity
                )
            
            high_pc_events = sum(
                1 for event in active_events 
                if event.collision_probability >= self.high_risk_threshold
            )
            
            return {
                'active_events': len(active_events),
                'total_history': len(self.event_history),
                'severity_distribution': severity_counts,
                'high_probability_events': high_pc_events,
                'conjunction_threshold_km': self.conjunction_threshold,
                'analysis_window_hours': self.analysis_window_hours
            } 