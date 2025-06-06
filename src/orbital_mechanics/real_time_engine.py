"""
Real-time State Propagation Engine

High-performance real-time orbital propagation system capable of handling
1000+ satellites with 1 Hz update rates, time synchronization, and state caching.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .satellite_state import SatelliteState, StateVector
from .propagator import StatePropagator
from ..data_management.satellite_manager import SatelliteManager


class EngineState(Enum):
    """Real-time engine operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PropagationMetrics:
    """Performance metrics for the propagation engine."""
    satellites_processed: int = 0
    propagation_time_ms: float = 0.0
    update_frequency_hz: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StateCache:
    """Efficient caching system for satellite states."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 60.0):
        """
        Initialize state cache.
        
        Args:
            max_size: Maximum number of cached states
            ttl_seconds: Time-to-live for cached states
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # (state, timestamp)
        self.access_order = deque()
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[StateVector]:
        """Get cached state if valid."""
        with self.lock:
            if key in self.cache:
                state, timestamp = self.cache[key]
                
                # Check if cache entry is still valid
                if (datetime.now() - timestamp).total_seconds() <= self.ttl_seconds:
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.hit_count += 1
                    return state
                else:
                    # Expired entry
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, state: StateVector):
        """Cache a state."""
        with self.lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = (state, datetime.now())
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def invalidate(self, key: str):
        """Invalidate a cached entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': self.get_hit_rate(),
                'ttl_seconds': self.ttl_seconds
            }


class RealTimePropagationEngine:
    """
    High-performance real-time orbital propagation engine.
    
    Manages continuous propagation of satellite states with precise timing,
    parallel processing, and advanced caching mechanisms.
    """
    
    def __init__(self, satellite_manager: SatelliteManager, config: Dict = None):
        """
        Initialize the real-time propagation engine.
        
        Args:
            satellite_manager: Satellite management system
            config: Configuration dictionary
        """
        self.satellite_manager = satellite_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Engine state
        self.state = EngineState.STOPPED
        self.state_lock = threading.RLock()
        
        # Timing configuration
        self.target_frequency = self.config.get('update_frequency_hz', 1.0)
        self.time_step = 1.0 / self.target_frequency
        self.max_time_drift = self.config.get('max_time_drift_ms', 10.0)
        
        # Performance configuration
        self.max_satellites = self.config.get('max_satellites', 1000)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.num_worker_threads = self.config.get('num_worker_threads', 4)
        self.batch_size = self.config.get('batch_size', 50)
        
        # Propagation components
        self.propagator = StatePropagator(
            use_j2=self.config.get('use_j2_perturbation', True)
        )
        
        # State caching
        cache_config = self.config.get('cache', {})
        self.state_cache = StateCache(
            max_size=cache_config.get('max_size', 10000),
            ttl_seconds=cache_config.get('ttl_seconds', 60.0)
        )
        
        # Threading
        self.propagation_thread: Optional[threading.Thread] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.shutdown_event = threading.Event()
        
        # Timing and synchronization
        self.last_update_time = 0.0
        self.update_counter = 0
        self.drift_compensation = 0.0
        
        # Metrics and monitoring
        self.metrics_history = deque(maxlen=1000)
        self.metrics_lock = threading.Lock()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'state_changed': [],
            'propagation_complete': [],
            'error_occurred': [],
            'metrics_updated': []
        }
        
        self.logger.info("Real-time Propagation Engine initialized")
    
    def start(self) -> bool:
        """
        Start the real-time propagation engine.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self.state_lock:
            if self.state != EngineState.STOPPED:
                self.logger.warning(f"Cannot start engine in state: {self.state}")
                return False
            
            self.state = EngineState.STARTING
        
        try:
            # Initialize threading components
            self.shutdown_event.clear()
            self.executor = ThreadPoolExecutor(max_workers=self.num_worker_threads)
            
            # Start propagation thread
            self.propagation_thread = threading.Thread(
                target=self._propagation_loop,
                name="PropagationLoop",
                daemon=True
            )
            self.propagation_thread.start()
            
            # Initialize timing
            self.last_update_time = time.time()
            self.update_counter = 0
            self.drift_compensation = 0.0
            
            with self.state_lock:
                self.state = EngineState.RUNNING
            
            self._trigger_event('state_changed', {'new_state': self.state})
            self.logger.info("Real-time Propagation Engine started")
            return True
            
        except Exception as e:
            with self.state_lock:
                self.state = EngineState.ERROR
            self.logger.error(f"Failed to start propagation engine: {e}")
            self._trigger_event('error_occurred', {'error': str(e)})
            return False
    
    def stop(self) -> bool:
        """
        Stop the real-time propagation engine.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self.state_lock:
            if self.state == EngineState.STOPPED:
                return True
            
            self.state = EngineState.STOPPING
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for propagation thread to finish
            if self.propagation_thread and self.propagation_thread.is_alive():
                self.propagation_thread.join(timeout=5.0)
                if self.propagation_thread.is_alive():
                    self.logger.warning("Propagation thread did not shut down gracefully")
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
            
            with self.state_lock:
                self.state = EngineState.STOPPED
            
            self._trigger_event('state_changed', {'new_state': self.state})
            self.logger.info("Real-time Propagation Engine stopped")
            return True
            
        except Exception as e:
            with self.state_lock:
                self.state = EngineState.ERROR
            self.logger.error(f"Error stopping propagation engine: {e}")
            return False
    
    def pause(self) -> bool:
        """Pause the propagation engine."""
        with self.state_lock:
            if self.state == EngineState.RUNNING:
                self.state = EngineState.PAUSED
                self._trigger_event('state_changed', {'new_state': self.state})
                self.logger.info("Propagation engine paused")
                return True
            return False
    
    def resume(self) -> bool:
        """Resume the propagation engine."""
        with self.state_lock:
            if self.state == EngineState.PAUSED:
                self.state = EngineState.RUNNING
                self.last_update_time = time.time()  # Reset timing
                self._trigger_event('state_changed', {'new_state': self.state})
                self.logger.info("Propagation engine resumed")
                return True
            return False
    
    def get_state(self) -> EngineState:
        """Get current engine state."""
        with self.state_lock:
            return self.state
    
    def _propagation_loop(self):
        """Main propagation loop running in separate thread."""
        self.logger.debug("Propagation loop started")
        
        while not self.shutdown_event.is_set():
            try:
                loop_start_time = time.time()
                
                # Check if engine is paused
                with self.state_lock:
                    if self.state == EngineState.PAUSED:
                        time.sleep(0.1)
                        continue
                    elif self.state != EngineState.RUNNING:
                        break
                
                # Perform propagation update
                metrics = self._perform_propagation_update()
                
                # Record metrics
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                
                # Trigger metrics callback
                self._trigger_event('metrics_updated', {'metrics': metrics})
                
                # Calculate timing for next update
                loop_duration = time.time() - loop_start_time
                self._handle_timing(loop_duration)
                
            except Exception as e:
                self.logger.error(f"Error in propagation loop: {e}")
                self._trigger_event('error_occurred', {'error': str(e)})
                
                # Brief pause before retrying
                time.sleep(0.1)
        
        self.logger.debug("Propagation loop ended")
    
    def _perform_propagation_update(self) -> PropagationMetrics:
        """Perform a single propagation update cycle."""
        update_start_time = time.time()
        error_count = 0
        
        # Get all active satellites
        satellites = self.satellite_manager.fleet.satellites
        active_satellites = [
            sat for sat in satellites.values()
            if sat.operational_status == "active"
        ]
        
        if not active_satellites:
            return PropagationMetrics(
                satellites_processed=0,
                propagation_time_ms=0.0,
                update_frequency_hz=0.0,
                cache_hit_rate=self.state_cache.get_hit_rate(),
                error_count=0
            )
        
        # Limit to maximum satellites
        if len(active_satellites) > self.max_satellites:
            active_satellites = active_satellites[:self.max_satellites]
            self.logger.warning(f"Limited propagation to {self.max_satellites} satellites")
        
        # Propagate satellites
        if self.parallel_processing and len(active_satellites) > self.batch_size:
            # Parallel processing for large fleets
            propagated_count, errors = self._propagate_parallel(active_satellites)
        else:
            # Sequential processing for smaller fleets
            propagated_count, errors = self._propagate_sequential(active_satellites)
        
        error_count = len(errors)
        
        # Calculate metrics
        propagation_duration = (time.time() - update_start_time) * 1000.0  # ms
        
        # Calculate frequency from recent updates
        current_time = time.time()
        if self.last_update_time > 0:
            actual_interval = current_time - self.last_update_time
            actual_frequency = 1.0 / actual_interval if actual_interval > 0 else 0.0
        else:
            actual_frequency = 0.0
        
        self.last_update_time = current_time
        self.update_counter += 1
        
        metrics = PropagationMetrics(
            satellites_processed=propagated_count,
            propagation_time_ms=propagation_duration,
            update_frequency_hz=actual_frequency,
            cache_hit_rate=self.state_cache.get_hit_rate(),
            error_count=error_count
        )
        
        # Log errors if any
        if errors:
            self.logger.warning(f"Propagation errors for {error_count} satellites")
            for sat_id, error in errors[:5]:  # Log first 5 errors
                self.logger.debug(f"Satellite {sat_id}: {error}")
        
        # Trigger propagation complete event
        self._trigger_event('propagation_complete', {
            'satellites_processed': propagated_count,
            'duration_ms': propagation_duration,
            'errors': errors
        })
        
        return metrics
    
    def _propagate_sequential(self, satellites: List[SatelliteState]) -> tuple[int, List[tuple]]:
        """Propagate satellites sequentially."""
        propagated_count = 0
        errors = []
        
        for satellite in satellites:
            try:
                # Check cache first
                cache_key = f"{satellite.satellite_id}_{self.update_counter}"
                cached_state = self.state_cache.get(cache_key)
                
                if cached_state is None:
                    # Propagate state
                    new_state = self.propagator.propagate_state(satellite, self.time_step)
                    
                    # Cache the result
                    self.state_cache.put(cache_key, new_state)
                    
                    # Update satellite
                    satellite.update_state_vector(new_state)
                else:
                    # Use cached state
                    satellite.update_state_vector(cached_state)
                
                propagated_count += 1
                
            except Exception as e:
                errors.append((satellite.satellite_id, str(e)))
        
        return propagated_count, errors
    
    def _propagate_parallel(self, satellites: List[SatelliteState]) -> tuple[int, List[tuple]]:
        """Propagate satellites in parallel using thread pool."""
        propagated_count = 0
        errors = []
        
        # Split satellites into batches
        batches = [
            satellites[i:i + self.batch_size]
            for i in range(0, len(satellites), self.batch_size)
        ]
        
        # Submit batch jobs
        futures = []
        for batch in batches:
            future = self.executor.submit(self._propagate_batch, batch)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_count, batch_errors = future.result()
                propagated_count += batch_count
                errors.extend(batch_errors)
            except Exception as e:
                self.logger.error(f"Batch propagation error: {e}")
                errors.append(("batch_error", str(e)))
        
        return propagated_count, errors
    
    def _propagate_batch(self, satellites: List[SatelliteState]) -> tuple[int, List[tuple]]:
        """Propagate a batch of satellites."""
        return self._propagate_sequential(satellites)
    
    def _handle_timing(self, loop_duration: float):
        """Handle timing and sleep to maintain target frequency."""
        target_interval = 1.0 / self.target_frequency
        sleep_time = target_interval - loop_duration - self.drift_compensation
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
            # Calculate actual interval and drift
            actual_interval = time.time() - (self.last_update_time or time.time())
            drift = actual_interval - target_interval
            
            # Apply drift compensation (PID-like control)
            self.drift_compensation += drift * 0.1  # Proportional term
            
            # Limit compensation to prevent overcorrection
            max_compensation = target_interval * 0.1
            self.drift_compensation = max(-max_compensation, 
                                        min(max_compensation, self.drift_compensation))
        else:
            # Running behind schedule
            self.logger.debug(f"Propagation cycle overrun: {-sleep_time:.3f}s")
            self.drift_compensation = 0.0
    
    def get_current_metrics(self) -> Optional[PropagationMetrics]:
        """Get the most recent propagation metrics."""
        with self.metrics_lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, count: int = 100) -> List[PropagationMetrics]:
        """Get recent metrics history."""
        with self.metrics_lock:
            return list(self.metrics_history)[-count:]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.metrics_lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 cycles
            
            # Calculate statistics
            frequencies = [m.update_frequency_hz for m in recent_metrics]
            durations = [m.propagation_time_ms for m in recent_metrics]
            cache_rates = [m.cache_hit_rate for m in recent_metrics]
            satellite_counts = [m.satellites_processed for m in recent_metrics]
            
            return {
                'engine_state': self.state.value,
                'total_updates': self.update_counter,
                'target_frequency_hz': self.target_frequency,
                'frequency_stats': {
                    'avg_hz': np.mean(frequencies) if frequencies else 0,
                    'min_hz': np.min(frequencies) if frequencies else 0,
                    'max_hz': np.max(frequencies) if frequencies else 0,
                    'std_hz': np.std(frequencies) if frequencies else 0
                },
                'duration_stats': {
                    'avg_ms': np.mean(durations) if durations else 0,
                    'min_ms': np.min(durations) if durations else 0,
                    'max_ms': np.max(durations) if durations else 0,
                    'std_ms': np.std(durations) if durations else 0
                },
                'cache_stats': self.state_cache.get_stats(),
                'avg_cache_hit_rate': np.mean(cache_rates) if cache_rates else 0,
                'avg_satellites_processed': np.mean(satellite_counts) if satellite_counts else 0,
                'drift_compensation': self.drift_compensation
            }
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register an event callback."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def _trigger_event(self, event_type: str, event_data: Dict):
        """Trigger event callbacks."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")
    
    def force_cache_clear(self):
        """Force clear the state cache."""
        self.state_cache.clear()
        self.logger.info("State cache cleared")
    
    def set_target_frequency(self, frequency_hz: float):
        """Change the target update frequency."""
        if frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        
        self.target_frequency = frequency_hz
        self.time_step = 1.0 / frequency_hz
        self.drift_compensation = 0.0  # Reset drift compensation
        
        self.logger.info(f"Target frequency changed to {frequency_hz} Hz")
    
    def get_satellite_states_snapshot(self) -> Dict[str, StateVector]:
        """Get a snapshot of all current satellite states."""
        satellites = self.satellite_manager.fleet.satellites
        return {
            sat_id: sat.state_vector for sat_id, sat in satellites.items()
            if sat.state_vector is not None
        } 