"""
STM Orbital Mechanics Engine

Main orchestrator for orbital mechanics functionality in the STM Digital Twin.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from .satellite_state import SatelliteState, OrbitalElements, StateVector
from .propagator import StatePropagator, J2Propagator, KeplerianPropagator
from .coordinate_systems import CoordinateTransformer


class STMOrbitEngine:
    """
    Main orbital mechanics engine for the STM Digital Twin system.
    
    Manages satellite states, propagation, and coordinate transformations.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the orbital mechanics engine.
        
        Args:
            config: Configuration dictionary with orbital mechanics settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        use_j2 = self.config.get('use_j2_perturbation', True)
        self.propagator = StatePropagator(use_j2=use_j2)
        self.coordinate_transformer = CoordinateTransformer()
        
        # Satellite management
        self.satellites: Dict[str, SatelliteState] = {}
        self.lock = threading.RLock()
        
        # Performance settings
        self.max_satellites = self.config.get('max_satellites', 1000)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.num_worker_threads = self.config.get('num_worker_threads', 4)
        
        # State caching
        self.state_cache = {}
        self.cache_duration = self.config.get('cache_duration_seconds', 60)
        
        self.logger.info("STM Orbital Mechanics Engine initialized")
    
    def add_satellite(self, satellite_state: SatelliteState) -> bool:
        """
        Add a satellite to the tracking system.
        
        Args:
            satellite_state: Satellite state to add
            
        Returns:
            True if successful, False if satellite limit reached
        """
        with self.lock:
            if len(self.satellites) >= self.max_satellites:
                self.logger.warning(f"Satellite limit ({self.max_satellites}) reached")
                return False
            
            self.satellites[satellite_state.satellite_id] = satellite_state
            self.logger.debug(f"Added satellite {satellite_state.satellite_id}")
            return True
    
    def remove_satellite(self, satellite_id: str) -> bool:
        """
        Remove a satellite from the tracking system.
        
        Args:
            satellite_id: ID of satellite to remove
            
        Returns:
            True if satellite was removed, False if not found
        """
        with self.lock:
            if satellite_id in self.satellites:
                del self.satellites[satellite_id]
                # Clean up cache
                if satellite_id in self.state_cache:
                    del self.state_cache[satellite_id]
                self.logger.debug(f"Removed satellite {satellite_id}")
                return True
            return False
    
    def get_satellite(self, satellite_id: str) -> Optional[SatelliteState]:
        """Get satellite state by ID."""
        with self.lock:
            return self.satellites.get(satellite_id)
    
    def get_all_satellites(self) -> Dict[str, SatelliteState]:
        """Get all satellite states."""
        with self.lock:
            return self.satellites.copy()
    
    def propagate_all(self, dt: float) -> Dict[str, StateVector]:
        """
        Propagate all satellites forward in time.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Dictionary mapping satellite IDs to new state vectors
        """
        start_time = datetime.now()
        
        with self.lock:
            satellite_list = list(self.satellites.values())
        
        if self.parallel_processing and len(satellite_list) > 10:
            # Use parallel processing for large satellite counts
            results = self._propagate_parallel(satellite_list, dt)
        else:
            # Sequential processing for small counts
            results = self.propagator.batch_propagate(satellite_list, dt)
        
        # Update satellite states
        with self.lock:
            for sat_id, new_state in results.items():
                if sat_id in self.satellites and new_state is not None:
                    self.satellites[sat_id].update_state_vector(new_state)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.debug(f"Propagated {len(results)} satellites in {elapsed:.3f}s")
        
        return results
    
    def _propagate_parallel(self, satellite_list: List[SatelliteState], dt: float) -> Dict[str, StateVector]:
        """Propagate satellites using parallel processing."""
        results = {}
        
        # Split satellites into chunks for parallel processing
        chunk_size = max(1, len(satellite_list) // self.num_worker_threads)
        chunks = [satellite_list[i:i + chunk_size] for i in range(0, len(satellite_list), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.num_worker_threads) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self.propagator.batch_propagate, chunk, dt)
                futures.append(future)
            
            # Collect results
            for future in futures:
                chunk_results = future.result()
                results.update(chunk_results)
        
        return results
    
    def propagate_single(self, satellite_id: str, dt: float) -> Optional[StateVector]:
        """
        Propagate a single satellite forward in time.
        
        Args:
            satellite_id: ID of satellite to propagate
            dt: Time step in seconds
            
        Returns:
            New state vector or None if satellite not found
        """
        with self.lock:
            satellite = self.satellites.get(satellite_id)
        
        if not satellite:
            return None
        
        try:
            new_state = self.propagator.propagate_state(satellite, dt)
            
            # Update satellite state
            with self.lock:
                satellite.update_state_vector(new_state)
            
            return new_state
        except Exception as e:
            self.logger.error(f"Error propagating satellite {satellite_id}: {e}")
            return None
    
    def get_positions_at_time(self, timestamp: datetime) -> Dict[str, np.ndarray]:
        """
        Get positions of all satellites at a specific time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Dictionary mapping satellite IDs to position vectors
        """
        positions = {}
        
        with self.lock:
            for sat_id, satellite in self.satellites.items():
                state = satellite.get_state_at_time(timestamp)
                if state:
                    positions[sat_id] = state.position
        
        return positions
    
    def calculate_ground_tracks(self, timestamp: datetime) -> Dict[str, Tuple[float, float]]:
        """
        Calculate ground tracks for all satellites at a specific time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Dictionary mapping satellite IDs to (latitude, longitude) tuples
        """
        ground_tracks = {}
        
        positions = self.get_positions_at_time(timestamp)
        for sat_id, position in positions.items():
            try:
                lat, lon = self.coordinate_transformer.calculate_ground_track(position, timestamp)
                ground_tracks[sat_id] = (lat, lon)
            except Exception as e:
                self.logger.error(f"Error calculating ground track for {sat_id}: {e}")
        
        return ground_tracks
    
    def calculate_relative_states(self, sat_id1: str, sat_id2: str) -> Optional[Dict]:
        """
        Calculate relative state between two satellites.
        
        Args:
            sat_id1: ID of first satellite
            sat_id2: ID of second satellite
            
        Returns:
            Dictionary with relative position, velocity, and distance
        """
        with self.lock:
            sat1 = self.satellites.get(sat_id1)
            sat2 = self.satellites.get(sat_id2)
        
        if not sat1 or not sat2 or not sat1.state_vector or not sat2.state_vector:
            return None
        
        state1 = sat1.state_vector
        state2 = sat2.state_vector
        
        # Calculate relative vectors
        rel_position = state1.position - state2.position
        rel_velocity = state1.velocity - state2.velocity
        distance = np.linalg.norm(rel_position)
        relative_speed = np.linalg.norm(rel_velocity)
        
        return {
            'satellite_1': sat_id1,
            'satellite_2': sat_id2,
            'relative_position': rel_position,
            'relative_velocity': rel_velocity,
            'distance_km': distance,
            'relative_speed_km_s': relative_speed,
            'timestamp': state1.timestamp
        }
    
    def get_satellites_in_range(self, center_position: np.ndarray, max_range_km: float) -> List[str]:
        """
        Get satellites within a specified range of a center position.
        
        Args:
            center_position: Center position vector in km
            max_range_km: Maximum range in km
            
        Returns:
            List of satellite IDs within range
        """
        satellites_in_range = []
        
        with self.lock:
            for sat_id, satellite in self.satellites.items():
                if satellite.state_vector:
                    distance = np.linalg.norm(satellite.state_vector.position - center_position)
                    if distance <= max_range_km:
                        satellites_in_range.append(sat_id)
        
        return satellites_in_range
    
    def update_from_tle(self, satellite_id: str, tle_line1: str, tle_line2: str) -> bool:
        """
        Update satellite orbital elements from TLE data.
        
        Args:
            satellite_id: ID of satellite to update
            tle_line1: First line of TLE
            tle_line2: Second line of TLE
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse TLE (simplified - in practice, use a proper TLE parser)
            # This is a placeholder for TLE parsing functionality
            self.logger.warning("TLE parsing not yet implemented")
            return False
        except Exception as e:
            self.logger.error(f"Error updating satellite {satellite_id} from TLE: {e}")
            return False
    
    def get_system_statistics(self) -> Dict:
        """Get statistics about the orbital mechanics system."""
        with self.lock:
            active_satellites = sum(1 for sat in self.satellites.values() 
                                  if sat.operational_status == "active")
            
            altitudes = [sat.current_altitude for sat in self.satellites.values() 
                        if sat.current_altitude is not None]
            
            return {
                'total_satellites': len(self.satellites),
                'active_satellites': active_satellites,
                'inactive_satellites': len(self.satellites) - active_satellites,
                'min_altitude_km': min(altitudes) if altitudes else None,
                'max_altitude_km': max(altitudes) if altitudes else None,
                'avg_altitude_km': np.mean(altitudes) if altitudes else None,
                'cache_size': len(self.state_cache),
                'propagator_type': 'J2' if self.propagator.use_j2 else 'Keplerian'
            }
    
    def cleanup_old_cache(self):
        """Remove old entries from the state cache."""
        current_time = datetime.now()
        
        with self.lock:
            expired_keys = []
            for key, (timestamp, _) in self.state_cache.items():
                if (current_time - timestamp).total_seconds() > self.cache_duration:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.state_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries") 