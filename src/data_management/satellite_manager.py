"""
Satellite Fleet Management System

Provides efficient storage, retrieval, and management of satellite objects
with thread-safe operations for concurrent access.
"""

import sys
import os
import logging
import threading
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

# Handle both relative and absolute imports
try:
    from ..orbital_mechanics.satellite_state import SatelliteState, OrbitalElements, StateVector
except ImportError:
    # If relative import fails, try absolute import for when running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from orbital_mechanics.satellite_state import SatelliteState, OrbitalElements, StateVector


@dataclass
class SatelliteMetadata:
    """Extended metadata for satellite management."""
    
    satellite_id: str
    name: str = ""
    owner: str = ""
    mission_type: str = ""
    launch_date: Optional[datetime] = None
    operational_criticality: int = 5  # 1-10 scale
    replacement_cost: float = 100_000_000  # USD
    fuel_capacity: float = 100.0  # kg
    fuel_remaining: float = 100.0  # kg
    power_generation: float = 1000.0  # Watts
    communication_frequency: float = 2.4  # GHz
    mass: float = 1000.0  # kg
    cross_sectional_area: float = 10.0  # m²
    operational_status: str = "active"
    operator: str = ""
    tags: Set[str] = field(default_factory=set)
    notes: str = ""
    last_contact: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate metadata values."""
        if not 1 <= self.operational_criticality <= 10:
            raise ValueError("Operational criticality must be between 1 and 10")
        if self.fuel_remaining > self.fuel_capacity:
            raise ValueError("Fuel remaining cannot exceed fuel capacity")


class SatelliteFleet:
    """
    Container for managing a fleet of satellites with advanced querying capabilities.
    """
    
    def __init__(self):
        """Initialize satellite fleet."""
        self.satellites: Dict[str, SatelliteState] = {}
        self.metadata: Dict[str, SatelliteMetadata] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Indexing for fast queries
        self._indices = {
            'by_owner': {},
            'by_mission_type': {},
            'by_criticality': {},
            'by_altitude_range': {},
            'by_operational_status': {}
        }
    
    def add_satellite(self, satellite_state: SatelliteState, 
                     metadata: Optional[SatelliteMetadata] = None) -> bool:
        """
        Add a satellite to the fleet.
        
        Args:
            satellite_state: Satellite state object
            metadata: Optional metadata object
            
        Returns:
            True if successful, False if satellite already exists
        """
        with self.lock:
            if satellite_state.satellite_id in self.satellites:
                self.logger.warning(f"Satellite {satellite_state.satellite_id} already exists")
                return False
            
            self.satellites[satellite_state.satellite_id] = satellite_state
            
            if metadata is None:
                metadata = SatelliteMetadata(satellite_id=satellite_state.satellite_id)
            self.metadata[satellite_state.satellite_id] = metadata
            
            self._update_indices(satellite_state.satellite_id, metadata)
            self.logger.debug(f"Added satellite {satellite_state.satellite_id} to fleet")
            return True
    
    def remove_satellite(self, satellite_id: str) -> bool:
        """Remove a satellite from the fleet."""
        with self.lock:
            if satellite_id not in self.satellites:
                return False
            
            del self.satellites[satellite_id]
            metadata = self.metadata.pop(satellite_id, None)
            
            if metadata:
                self._remove_from_indices(satellite_id, metadata)
            
            self.logger.debug(f"Removed satellite {satellite_id} from fleet")
            return True
    
    def get_satellite(self, satellite_id: str) -> Optional[SatelliteState]:
        """Get satellite by ID."""
        with self.lock:
            return self.satellites.get(satellite_id)
    
    def get_metadata(self, satellite_id: str) -> Optional[SatelliteMetadata]:
        """Get satellite metadata by ID."""
        with self.lock:
            return self.metadata.get(satellite_id)
    
    def update_metadata(self, satellite_id: str, metadata: SatelliteMetadata) -> bool:
        """Update satellite metadata."""
        with self.lock:
            if satellite_id not in self.satellites:
                return False
            
            old_metadata = self.metadata.get(satellite_id)
            if old_metadata:
                self._remove_from_indices(satellite_id, old_metadata)
            
            self.metadata[satellite_id] = metadata
            self._update_indices(satellite_id, metadata)
            return True
    
    def query_by_owner(self, owner: str) -> List[str]:
        """Get satellites by owner."""
        with self.lock:
            return self._indices['by_owner'].get(owner, []).copy()
    
    def query_by_mission_type(self, mission_type: str) -> List[str]:
        """Get satellites by mission type."""
        with self.lock:
            return self._indices['by_mission_type'].get(mission_type, []).copy()
    
    def query_by_criticality(self, min_criticality: int, max_criticality: int = 10) -> List[str]:
        """Get satellites by criticality range."""
        result = []
        with self.lock:
            for criticality in range(min_criticality, max_criticality + 1):
                result.extend(self._indices['by_criticality'].get(criticality, []))
        return result
    
    def query_by_altitude_range(self, min_alt: float, max_alt: float) -> List[str]:
        """Get satellites by altitude range."""
        result = []
        with self.lock:
            for sat_id, satellite in self.satellites.items():
                if satellite.current_altitude is not None:
                    if min_alt <= satellite.current_altitude <= max_alt:
                        result.append(sat_id)
        return result
    
    def query_by_operational_status(self, status: str) -> List[str]:
        """Get satellites by operational status."""
        with self.lock:
            return self._indices['by_operational_status'].get(status, []).copy()
    
    def query_active_satellites(self) -> List[str]:
        """Get all active satellites."""
        return self.query_by_operational_status("active")
    
    def query_critical_satellites(self, threshold: int = 8) -> List[str]:
        """Get satellites with high criticality."""
        return self.query_by_criticality(threshold, 10)
    
    def query_low_fuel_satellites(self, threshold_percent: float = 20.0) -> List[str]:
        """Get satellites with low fuel."""
        result = []
        with self.lock:
            for sat_id, metadata in self.metadata.items():
                fuel_percent = (metadata.fuel_remaining / metadata.fuel_capacity) * 100
                if fuel_percent <= threshold_percent:
                    result.append(sat_id)
        return result
    
    def get_fleet_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fleet statistics."""
        with self.lock:
            total = len(self.satellites)
            if total == 0:
                return {'total_satellites': 0}
            
            # Status distribution
            status_counts = {}
            for satellite in self.satellites.values():
                status = satellite.operational_status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Criticality distribution
            criticality_counts = {}
            for metadata in self.metadata.values():
                crit = metadata.operational_criticality
                criticality_counts[crit] = criticality_counts.get(crit, 0) + 1
            
            # Altitude statistics
            altitudes = [sat.current_altitude for sat in self.satellites.values() 
                        if sat.current_altitude is not None]
            
            # Fuel statistics
            fuel_levels = [(meta.fuel_remaining / meta.fuel_capacity) * 100 
                          for meta in self.metadata.values()]
            
            return {
                'total_satellites': total,
                'status_distribution': status_counts,
                'criticality_distribution': criticality_counts,
                'altitude_stats': {
                    'min_km': min(altitudes) if altitudes else None,
                    'max_km': max(altitudes) if altitudes else None,
                    'avg_km': sum(altitudes) / len(altitudes) if altitudes else None
                },
                'fuel_stats': {
                    'avg_percent': sum(fuel_levels) / len(fuel_levels) if fuel_levels else None,
                    'min_percent': min(fuel_levels) if fuel_levels else None,
                    'low_fuel_count': len([f for f in fuel_levels if f < 20])
                }
            }
    
    def _update_indices(self, satellite_id: str, metadata: SatelliteMetadata):
        """Update search indices."""
        # Owner index
        if metadata.owner:
            if metadata.owner not in self._indices['by_owner']:
                self._indices['by_owner'][metadata.owner] = []
            self._indices['by_owner'][metadata.owner].append(satellite_id)
        
        # Mission type index
        if metadata.mission_type:
            if metadata.mission_type not in self._indices['by_mission_type']:
                self._indices['by_mission_type'][metadata.mission_type] = []
            self._indices['by_mission_type'][metadata.mission_type].append(satellite_id)
        
        # Criticality index
        crit = metadata.operational_criticality
        if crit not in self._indices['by_criticality']:
            self._indices['by_criticality'][crit] = []
        self._indices['by_criticality'][crit].append(satellite_id)
        
        # Operational status index
        if satellite_id in self.satellites:
            status = self.satellites[satellite_id].operational_status
            if status not in self._indices['by_operational_status']:
                self._indices['by_operational_status'][status] = []
            self._indices['by_operational_status'][status].append(satellite_id)
    
    def _remove_from_indices(self, satellite_id: str, metadata: SatelliteMetadata):
        """Remove satellite from search indices."""
        # Remove from all indices
        for index_dict in self._indices.values():
            for sat_list in index_dict.values():
                if satellite_id in sat_list:
                    sat_list.remove(satellite_id)


class SatelliteManager:
    """
    High-level satellite management system with persistence and advanced features.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize satellite manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize fleet
        self.fleet = SatelliteFleet()
        
        # Persistence settings
        self.persistence_enabled = self.config.get('persistence_enabled', True)
        self.data_directory = Path(self.config.get('data_directory', 'data/satellites'))
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Auto-save settings
        self.auto_save_enabled = self.config.get('auto_save_enabled', True)
        self.auto_save_interval = self.config.get('auto_save_interval_minutes', 5)
        self.last_save_time = datetime.now()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'satellite_added': [],
            'satellite_removed': [],
            'satellite_updated': [],
            'low_fuel_detected': [],
            'status_changed': []
        }
        
        # Load existing data
        if self.persistence_enabled:
            self.load_fleet_data()
        
        self.logger.info("Satellite Manager initialized")
    
    def add_satellite(self, satellite_state: SatelliteState, 
                     metadata: Optional[SatelliteMetadata] = None,
                     auto_save: bool = True) -> bool:
        """
        Add a satellite with optional metadata.
        
        Args:
            satellite_state: Satellite state object
            metadata: Optional metadata
            auto_save: Whether to trigger auto-save
            
        Returns:
            True if successful
        """
        with self.lock:
            success = self.fleet.add_satellite(satellite_state, metadata)
            
            if success:
                self._trigger_event('satellite_added', {
                    'satellite_id': satellite_state.satellite_id,
                    'timestamp': datetime.now()
                })
                
                if auto_save and self.auto_save_enabled:
                    self._check_auto_save()
            
            return success
    
    def remove_satellite(self, satellite_id: str, auto_save: bool = True) -> bool:
        """Remove a satellite from management."""
        with self.lock:
            success = self.fleet.remove_satellite(satellite_id)
            
            if success:
                self._trigger_event('satellite_removed', {
                    'satellite_id': satellite_id,
                    'timestamp': datetime.now()
                })
                
                if auto_save and self.auto_save_enabled:
                    self._check_auto_save()
            
            return success
    
    def update_satellite_status(self, satellite_id: str, new_status: str, 
                              auto_save: bool = True) -> bool:
        """Update satellite operational status."""
        with self.lock:
            satellite = self.fleet.get_satellite(satellite_id)
            if not satellite:
                return False
            
            old_status = satellite.operational_status
            satellite.operational_status = new_status
            
            self._trigger_event('status_changed', {
                'satellite_id': satellite_id,
                'old_status': old_status,
                'new_status': new_status,
                'timestamp': datetime.now()
            })
            
            if auto_save and self.auto_save_enabled:
                self._check_auto_save()
            
            return True
    
    def bulk_update_fuel(self, fuel_updates: Dict[str, float]) -> int:
        """
        Update fuel levels for multiple satellites.
        
        Args:
            fuel_updates: Dictionary mapping satellite IDs to new fuel levels
            
        Returns:
            Number of satellites successfully updated
        """
        updated_count = 0
        low_fuel_threshold = self.config.get('low_fuel_threshold_percent', 20.0)
        
        with self.lock:
            for sat_id, new_fuel in fuel_updates.items():
                metadata = self.fleet.get_metadata(sat_id)
                if metadata:
                    old_fuel = metadata.fuel_remaining
                    metadata.fuel_remaining = max(0, min(new_fuel, metadata.fuel_capacity))
                    updated_count += 1
                    
                    # Check for low fuel
                    fuel_percent = (metadata.fuel_remaining / metadata.fuel_capacity) * 100
                    if fuel_percent <= low_fuel_threshold:
                        self._trigger_event('low_fuel_detected', {
                            'satellite_id': sat_id,
                            'fuel_percent': fuel_percent,
                            'timestamp': datetime.now()
                        })
        
        if updated_count > 0 and self.auto_save_enabled:
            self._check_auto_save()
        
        return updated_count
    
    def get_satellites_by_query(self, query_params: Dict[str, Any]) -> List[str]:
        """
        Advanced satellite querying with multiple criteria.
        
        Args:
            query_params: Dictionary with query parameters
            
        Returns:
            List of satellite IDs matching criteria
        """
        with self.lock:
            result_sets = []
            
            # Owner filter
            if 'owner' in query_params:
                result_sets.append(set(self.fleet.query_by_owner(query_params['owner'])))
            
            # Mission type filter
            if 'mission_type' in query_params:
                result_sets.append(set(self.fleet.query_by_mission_type(query_params['mission_type'])))
            
            # Criticality range filter
            if 'min_criticality' in query_params:
                min_crit = query_params['min_criticality']
                max_crit = query_params.get('max_criticality', 10)
                result_sets.append(set(self.fleet.query_by_criticality(min_crit, max_crit)))
            
            # Altitude range filter
            if 'min_altitude' in query_params and 'max_altitude' in query_params:
                min_alt = query_params['min_altitude']
                max_alt = query_params['max_altitude']
                result_sets.append(set(self.fleet.query_by_altitude_range(min_alt, max_alt)))
            
            # Status filter
            if 'status' in query_params:
                result_sets.append(set(self.fleet.query_by_operational_status(query_params['status'])))
            
            # Intersect all result sets
            if result_sets:
                final_result = result_sets[0]
                for result_set in result_sets[1:]:
                    final_result = final_result.intersection(result_set)
                return list(final_result)
            else:
                # No filters, return all satellites
                return list(self.fleet.satellites.keys())
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register a callback for satellite events."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def save_fleet_data(self, filename: str = None):
        """Save fleet data to disk."""
        if not self.persistence_enabled:
            return
        
        if filename is None:
            filename = f"fleet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_directory / filename
        
        with self.lock:
            # Prepare data for serialization
            fleet_data = {
                'satellites': {},
                'metadata': {},
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            for sat_id, satellite in self.fleet.satellites.items():
                fleet_data['satellites'][sat_id] = satellite.to_dict()
                
                metadata = self.fleet.metadata.get(sat_id)
                if metadata:
                    meta_dict = {
                        'satellite_id': metadata.satellite_id,
                        'name': metadata.name,
                        'owner': metadata.owner,
                        'mission_type': metadata.mission_type,
                        'launch_date': metadata.launch_date.isoformat() if metadata.launch_date else None,
                        'operational_criticality': metadata.operational_criticality,
                        'replacement_cost': metadata.replacement_cost,
                        'fuel_capacity': metadata.fuel_capacity,
                        'fuel_remaining': metadata.fuel_remaining,
                        'power_generation': metadata.power_generation,
                        'communication_frequency': metadata.communication_frequency,
                        'mass': metadata.mass,
                        'cross_sectional_area': metadata.cross_sectional_area,
                        'operational_status': metadata.operational_status,
                        'operator': metadata.operator,
                        'tags': list(metadata.tags),
                        'notes': metadata.notes,
                        'last_contact': metadata.last_contact.isoformat() if metadata.last_contact else None
                    }
                    fleet_data['metadata'][sat_id] = meta_dict
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(fleet_data, f, indent=2)
        
        self.last_save_time = datetime.now()
        self.logger.info(f"Fleet data saved to {filepath}")
    
    def load_fleet_data(self, filename: str = None):
        """Load fleet data from disk."""
        if not self.persistence_enabled:
            return
        
        if filename is None:
            # Find the most recent fleet data file
            data_files = list(self.data_directory.glob("fleet_data_*.json"))
            if not data_files:
                self.logger.info("No existing fleet data found")
                return
            
            filepath = max(data_files, key=lambda f: f.stat().st_mtime)
        else:
            filepath = self.data_directory / filename
        
        if not filepath.exists():
            self.logger.warning(f"Fleet data file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                fleet_data = json.load(f)
            
            # TODO: Implement full data loading from JSON
            # This would require deserializing SatelliteState and SatelliteMetadata objects
            # For now, just log that we found the data
            
            satellite_count = len(fleet_data.get('satellites', {}))
            self.logger.info(f"Found fleet data with {satellite_count} satellites in {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading fleet data from {filepath}: {e}")
    
    def _trigger_event(self, event_type: str, event_data: Dict):
        """Trigger event callbacks."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")
    
    def _check_auto_save(self):
        """Check if auto-save should be triggered."""
        now = datetime.now()
        if (now - self.last_save_time).total_seconds() >= self.auto_save_interval * 60:
            self.save_fleet_data()
    
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get a comprehensive fleet summary."""
        with self.lock:
            stats = self.fleet.get_fleet_statistics()
            
            # Add manager-specific information
            stats.update({
                'persistence_enabled': self.persistence_enabled,
                'auto_save_enabled': self.auto_save_enabled,
                'last_save_time': self.last_save_time.isoformat(),
                'data_directory': str(self.data_directory),
                'event_callback_count': sum(len(callbacks) for callbacks in self.event_callbacks.values())
            })
            
            return stats 