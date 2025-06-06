"""
Spatial Indexing System for Conjunction Detection

Efficient 3D spatial indexing using KDTree for fast nearest neighbor searches
and conjunction detection in satellite constellations.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.spatial import KDTree
import threading
from datetime import datetime


@dataclass
class SpatialObject:
    """Represents a spatial object in the index."""
    object_id: str
    position: np.ndarray  # 3D position vector
    velocity: np.ndarray  # 3D velocity vector (optional)
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SpatialIndex(ABC):
    """Abstract base class for spatial indexing systems."""
    
    @abstractmethod
    def insert(self, obj: SpatialObject) -> bool:
        """Insert a spatial object into the index."""
        pass
    
    @abstractmethod
    def remove(self, object_id: str) -> bool:
        """Remove an object from the index."""
        pass
    
    @abstractmethod
    def update(self, obj: SpatialObject) -> bool:
        """Update an object's position in the index."""
        pass
    
    @abstractmethod
    def nearest_neighbors(self, position: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a position."""
        pass
    
    @abstractmethod
    def range_query(self, position: np.ndarray, radius: float) -> List[Tuple[str, float]]:
        """Find all objects within radius of position."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all objects from the index."""
        pass


class KDTreeSpatialIndex(SpatialIndex):
    """
    KDTree-based spatial index for efficient 3D proximity searches.
    
    Optimized for dynamic updates and fast nearest neighbor queries.
    """
    
    def __init__(self, rebuild_threshold: int = 1000):
        """
        Initialize KDTree spatial index.
        
        Args:
            rebuild_threshold: Number of updates before rebuilding the tree
        """
        self.rebuild_threshold = rebuild_threshold
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.objects: Dict[str, SpatialObject] = {}
        self.positions: np.ndarray = np.empty((0, 3))
        self.object_ids: List[str] = []
        
        # KDTree
        self.kdtree: Optional[KDTree] = None
        self.tree_valid = False
        
        # Update tracking
        self.update_count = 0
        self.pending_updates: Set[str] = set()
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.debug("KDTree Spatial Index initialized")
    
    def insert(self, obj: SpatialObject) -> bool:
        """Insert a spatial object into the index."""
        with self.lock:
            if obj.object_id in self.objects:
                # Update existing object
                return self.update(obj)
            
            # Add new object
            self.objects[obj.object_id] = obj
            self.object_ids.append(obj.object_id)
            
            # Add position
            if self.positions.size == 0:
                self.positions = obj.position.reshape(1, -1)
            else:
                self.positions = np.vstack([self.positions, obj.position])
            
            # Mark tree as invalid
            self.tree_valid = False
            self.update_count += 1
            
            # Rebuild tree if threshold reached
            if self.update_count >= self.rebuild_threshold:
                self._rebuild_tree()
            
            return True
    
    def remove(self, object_id: str) -> bool:
        """Remove an object from the index."""
        with self.lock:
            if object_id not in self.objects:
                return False
            
            # Find object index
            try:
                obj_index = self.object_ids.index(object_id)
            except ValueError:
                return False
            
            # Remove from data structures
            del self.objects[object_id]
            self.object_ids.pop(obj_index)
            self.positions = np.delete(self.positions, obj_index, axis=0)
            
            # Remove from pending updates
            self.pending_updates.discard(object_id)
            
            # Mark tree as invalid
            self.tree_valid = False
            self.update_count += 1
            
            return True
    
    def update(self, obj: SpatialObject) -> bool:
        """Update an object's position in the index."""
        with self.lock:
            if obj.object_id not in self.objects:
                return self.insert(obj)
            
            # Update object data
            self.objects[obj.object_id] = obj
            
            # Update position array
            try:
                obj_index = self.object_ids.index(obj.object_id)
                self.positions[obj_index] = obj.position
            except ValueError:
                return False
            
            # Mark for pending update
            self.pending_updates.add(obj.object_id)
            self.update_count += 1
            
            # Invalidate tree for major updates
            if len(self.pending_updates) > self.rebuild_threshold // 4:
                self.tree_valid = False
            
            # Rebuild tree if threshold reached
            if self.update_count >= self.rebuild_threshold:
                self._rebuild_tree()
            
            return True
    
    def nearest_neighbors(self, position: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a position."""
        with self.lock:
            if not self._ensure_tree_valid():
                return []
            
            if len(self.object_ids) == 0:
                return []
            
            # Limit k to available objects
            k = min(k, len(self.object_ids))
            
            # Query KDTree
            distances, indices = self.kdtree.query(position, k=k)
            
            # Handle single result case
            if k == 1:
                distances = [distances]
                indices = [indices]
            
            # Build result list
            results = []
            for dist, idx in zip(distances, indices):
                if idx < len(self.object_ids):
                    object_id = self.object_ids[idx]
                    results.append((object_id, float(dist)))
            
            return results
    
    def range_query(self, position: np.ndarray, radius: float) -> List[Tuple[str, float]]:
        """Find all objects within radius of position."""
        with self.lock:
            if not self._ensure_tree_valid():
                return []
            
            if len(self.object_ids) == 0:
                return []
            
            # Query KDTree
            indices = self.kdtree.query_ball_point(position, radius)
            
            # Build result list with distances
            results = []
            for idx in indices:
                if idx < len(self.object_ids):
                    object_id = self.object_ids[idx]
                    obj_position = self.positions[idx]
                    distance = np.linalg.norm(position - obj_position)
                    results.append((object_id, float(distance)))
            
            # Sort by distance
            results.sort(key=lambda x: x[1])
            
            return results
    
    def range_query_pairs(self, radius: float) -> List[Tuple[str, str, float]]:
        """Find all pairs of objects within radius of each other."""
        with self.lock:
            if not self._ensure_tree_valid():
                return []
            
            if len(self.object_ids) < 2:
                return []
            
            # Query KDTree for all pairs
            pairs = self.kdtree.query_pairs(radius)
            
            # Convert indices to object IDs with distances
            results = []
            for idx1, idx2 in pairs:
                if idx1 < len(self.object_ids) and idx2 < len(self.object_ids):
                    obj_id1 = self.object_ids[idx1]
                    obj_id2 = self.object_ids[idx2]
                    pos1 = self.positions[idx1]
                    pos2 = self.positions[idx2]
                    distance = np.linalg.norm(pos1 - pos2)
                    results.append((obj_id1, obj_id2, float(distance)))
            
            # Sort by distance
            results.sort(key=lambda x: x[2])
            
            return results
    
    def get_object(self, object_id: str) -> Optional[SpatialObject]:
        """Get object by ID."""
        with self.lock:
            return self.objects.get(object_id)
    
    def get_all_objects(self) -> Dict[str, SpatialObject]:
        """Get all objects in the index."""
        with self.lock:
            return self.objects.copy()
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        with self.lock:
            return {
                'object_count': len(self.objects),
                'tree_valid': self.tree_valid,
                'update_count': self.update_count,
                'pending_updates': len(self.pending_updates),
                'rebuild_threshold': self.rebuild_threshold,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def clear(self):
        """Clear all objects from the index."""
        with self.lock:
            self.objects.clear()
            self.object_ids.clear()
            self.positions = np.empty((0, 3))
            self.kdtree = None
            self.tree_valid = False
            self.update_count = 0
            self.pending_updates.clear()
    
    def force_rebuild(self):
        """Force rebuild of the KDTree."""
        with self.lock:
            self._rebuild_tree()
    
    def _ensure_tree_valid(self) -> bool:
        """Ensure the KDTree is valid and up-to-date."""
        if not self.tree_valid or self.kdtree is None:
            return self._rebuild_tree()
        return True
    
    def _rebuild_tree(self) -> bool:
        """Rebuild the KDTree from current data."""
        try:
            if len(self.object_ids) == 0:
                self.kdtree = None
                self.tree_valid = True
                return True
            
            # Build new KDTree
            self.kdtree = KDTree(self.positions)
            self.tree_valid = True
            self.update_count = 0
            self.pending_updates.clear()
            
            self.logger.debug(f"Rebuilt KDTree with {len(self.object_ids)} objects")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebuilding KDTree: {e}")
            self.tree_valid = False
            return False
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        base_size = 0
        
        # Objects dictionary
        base_size += len(self.objects) * 1000  # Rough estimate per object
        
        # Positions array
        if self.positions.size > 0:
            base_size += self.positions.nbytes
        
        # Object IDs list
        base_size += len(self.object_ids) * 100  # Rough estimate per ID
        
        # KDTree (rough estimate)
        if self.kdtree is not None:
            base_size += len(self.object_ids) * 200  # Rough estimate
        
        return base_size / (1024 * 1024)  # Convert to MB


class AdaptiveSpatialIndex(KDTreeSpatialIndex):
    """
    Adaptive spatial index that adjusts rebuild frequency based on query patterns.
    """
    
    def __init__(self, initial_rebuild_threshold: int = 1000):
        """Initialize adaptive spatial index."""
        super().__init__(rebuild_threshold=initial_rebuild_threshold)
        
        # Adaptive parameters
        self.query_count = 0
        self.rebuild_count = 0
        self.last_performance_check = datetime.now()
        self.performance_window = 60  # seconds
        
        # Performance tracking
        self.query_times = []
        self.rebuild_times = []
    
    def nearest_neighbors(self, position: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors with performance tracking."""
        start_time = datetime.now()
        result = super().nearest_neighbors(position, k)
        query_time = (datetime.now() - start_time).total_seconds()
        
        self.query_times.append(query_time)
        self.query_count += 1
        
        # Check if we should adjust rebuild threshold
        self._check_performance()
        
        return result
    
    def range_query(self, position: np.ndarray, radius: float) -> List[Tuple[str, float]]:
        """Find objects in range with performance tracking."""
        start_time = datetime.now()
        result = super().range_query(position, radius)
        query_time = (datetime.now() - start_time).total_seconds()
        
        self.query_times.append(query_time)
        self.query_count += 1
        
        # Check if we should adjust rebuild threshold
        self._check_performance()
        
        return result
    
    def _rebuild_tree(self) -> bool:
        """Rebuild tree with performance tracking."""
        start_time = datetime.now()
        result = super()._rebuild_tree()
        rebuild_time = (datetime.now() - start_time).total_seconds()
        
        self.rebuild_times.append(rebuild_time)
        self.rebuild_count += 1
        
        return result
    
    def _check_performance(self):
        """Check and adjust performance parameters."""
        now = datetime.now()
        
        # Only check periodically
        if (now - self.last_performance_check).total_seconds() < self.performance_window:
            return
        
        self.last_performance_check = now
        
        # Get recent performance data
        recent_queries = self.query_times[-100:] if self.query_times else []
        recent_rebuilds = self.rebuild_times[-10:] if self.rebuild_times else []
        
        if not recent_queries:
            return
        
        avg_query_time = np.mean(recent_queries)
        avg_rebuild_time = np.mean(recent_rebuilds) if recent_rebuilds else 0.1
        
        # Adjust rebuild threshold based on performance
        if avg_query_time > 0.01 and avg_rebuild_time < 0.5:
            # Queries are slow, rebuilds are fast - rebuild more often
            self.rebuild_threshold = max(500, int(self.rebuild_threshold * 0.8))
        elif avg_query_time < 0.001 and avg_rebuild_time > 1.0:
            # Queries are fast, rebuilds are slow - rebuild less often
            self.rebuild_threshold = min(5000, int(self.rebuild_threshold * 1.2))
        
        self.logger.debug(f"Adjusted rebuild threshold to {self.rebuild_threshold}")
        
        # Clear old performance data
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-500:]
        if len(self.rebuild_times) > 50:
            self.rebuild_times = self.rebuild_times[-25:] 