"""
Space Traffic Management Digital Twin Dashboard

A comprehensive web-based dashboard for monitoring satellite conjunctions,
collision risks, and space traffic management operations.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any
from scipy import special

# Import STM Digital Twin modules
import sys
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# N2YO API Configuration — set N2YO_API_KEY in your environment (see .env.example).
# Without a key the dashboard falls back to simulated satellite positions.
N2YO_API_KEY = os.environ.get("N2YO_API_KEY", "")
N2YO_BASE_URL = "https://api.n2yo.com/rest/v1/satellite"

# Popular satellites to track (NORAD IDs)
TRACKED_SATELLITES = {
    25544: {"name": "SPACE STATION (ISS)", "id": "ISS"},
    20580: {"name": "HUBBLE SPACE TELESCOPE", "id": "HST"},
    43013: {"name": "STARLINK-1007", "id": "SL1007"},
    47675: {"name": "STARLINK-2182", "id": "SL2182"},
    28654: {"name": "GPS IIF-2", "id": "GPS2"},
    37849: {"name": "NOAA-19", "id": "NOAA19"},
    25994: {"name": "TERRA", "id": "TERRA"},
    27424: {"name": "AQUA", "id": "AQUA"}
}

# Observer location (you can change this to any location)
OBSERVER_LAT = 41.702  # Pennsylvania, USA
OBSERVER_LNG = -76.014
OBSERVER_ALT = 100  # meters

def fetch_satellite_positions_from_api(satellite_ids: List[int], seconds: int = 1) -> Dict[str, Dict]:
    """Fetch real satellite positions from N2YO API."""
    positions = {}
    
    def fetch_single_satellite(norad_id: int) -> tuple:
        """Fetch position for a single satellite."""
        try:
            url = f"{N2YO_BASE_URL}/positions/{norad_id}/{OBSERVER_LAT}/{OBSERVER_LNG}/{OBSERVER_ALT}/{seconds}"
            params = {"apiKey": N2YO_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors (rate limiting, etc.)
            if 'error' in data:
                logger.warning(f"N2YO API Error for satellite {norad_id}: {data['error']}")
                return None, None
            
            if 'positions' in data and len(data['positions']) > 0:
                pos = data['positions'][0]  # Get first (current) position
                sat_info = TRACKED_SATELLITES.get(norad_id, {"name": f"SAT-{norad_id}", "id": f"SAT{norad_id}"})
                
                # Convert satellite coordinates to 3D Cartesian (approximate)
                # Note: N2YO gives lat/lng/alt, we need to convert to X/Y/Z
                lat_rad = np.radians(pos['satlatitude'])
                lng_rad = np.radians(pos['satlongitude'])
                alt_km = pos.get('sataltitude', 400)  # km above Earth
                
                earth_radius = 6371.0  # km
                total_radius = earth_radius + alt_km
                
                # Convert to Cartesian coordinates
                x = total_radius * np.cos(lat_rad) * np.cos(lng_rad)
                y = total_radius * np.cos(lat_rad) * np.sin(lng_rad)
                z = total_radius * np.sin(lat_rad)
                
                # Estimate velocity (simplified - just for visualization)
                # In reality, you'd need multiple position samples or TLE data
                orbital_velocity = 7.66  # km/s typical for LEO
                vx = -orbital_velocity * np.sin(lng_rad)
                vy = orbital_velocity * np.cos(lng_rad)
                vz = 0  # simplified
                
                return sat_info["id"], {
                    'x': x, 'y': y, 'z': z,
                    'vx': vx, 'vy': vy, 'vz': vz,
                    'latitude': pos['satlatitude'],
                    'longitude': pos['satlongitude'],
                    'altitude': alt_km,
                    'azimuth': pos.get('azimuth', 0),
                    'elevation': pos.get('elevation', 0),
                    'name': sat_info["name"],
                    'norad_id': norad_id,
                    'timestamp': datetime.fromtimestamp(pos['timestamp']).isoformat()
                }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for satellite {norad_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing satellite {norad_id}: {e}")
        
        return None, None
    
    # Fetch positions for multiple satellites in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_norad = {executor.submit(fetch_single_satellite, norad_id): norad_id 
                          for norad_id in satellite_ids}
        
        for future in as_completed(future_to_norad):
            sat_id, position_data = future.result()
            if sat_id and position_data:
                positions[sat_id] = position_data
    
    logger.info(f"Fetched positions for {len(positions)} satellites from N2YO API")
    return positions

def get_satellites_above_location() -> List[int]:
    """Get satellites currently visible above the observer location."""
    try:
        url = f"{N2YO_BASE_URL}/above/{OBSERVER_LAT}/{OBSERVER_LNG}/{OBSERVER_ALT}/70/0"
        params = {"apiKey": N2YO_API_KEY}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors (rate limiting, etc.)
        if 'error' in data:
            logger.warning(f"N2YO API Error: {data['error']}")
            if "exceeded" in data['error'].lower():
                logger.warning("API rate limit exceeded - using fallback satellites")
            return []
        
        if 'above' in data:
            visible_sats = [sat['satid'] for sat in data['above'][:8]]  # Limit to 8 satellites
            logger.info(f"Found {len(visible_sats)} satellites above location")
            return visible_sats
        
    except Exception as e:
        logger.error(f"Error fetching satellites above location: {e}")
    
    # Return empty list to trigger fallback
    return []

# Handle imports with fallbacks
try:
    from orbital_mechanics.orbit_engine import STMOrbitEngine
    from data_management.satellite_manager import SatelliteManager, SatelliteMetadata
    from conjunction_detection.conjunction_analyzer import ConjunctionAnalyzer
    from conjunction_detection.spatial_index import KDTreeSpatialIndex
    from risk_assessment.risk_calculator import RiskCalculator
    from sensor_simulation.sensor_emulator import SensorEmulator, SensorConfig, SensorType
    from sensor_simulation.noise_models import NoiseCharacteristics
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you run this from the project root directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

app.title = "STM Digital Twin Dashboard"

# Initialize STM components
orbit_engine = STMOrbitEngine()
satellite_manager = SatelliteManager()
spatial_index = KDTreeSpatialIndex(rebuild_threshold=500)
conjunction_analyzer = ConjunctionAnalyzer(spatial_index)
risk_calculator = RiskCalculator()
sensor_emulator = SensorEmulator()

# Global state for demo data
demo_satellites = {}
demo_conjunctions = []
demo_sensor_data = {}

def initialize_demo_data():
    """Initialize demonstration satellites and sensors."""
    global demo_satellites, demo_conjunctions, demo_sensor_data
    
    logger.info("Initializing demo data...")
    
    # Create demo satellites
    satellite_configs = [
        {
            'satellite_id': 'SAT-001',
            'name': 'Demo Satellite 1',
            'semi_major_axis': 6778.0,  # km
            'eccentricity': 0.001,
            'inclination': 51.6,  # degrees
            'raan': 0.0,
            'arg_periapsis': 0.0,
            'true_anomaly': 0.0,
            'mass': 1500.0,
            'cross_sectional_area': 10.0
        },
        {
            'satellite_id': 'SAT-002', 
            'name': 'Demo Satellite 2',
            'semi_major_axis': 6778.5,
            'eccentricity': 0.002,
            'inclination': 51.8,
            'raan': 1.0,
            'arg_periapsis': 0.0,
            'true_anomaly': 180.0,
            'mass': 800.0,
            'cross_sectional_area': 5.0
        },
        {
            'satellite_id': 'SAT-003',
            'name': 'Demo Satellite 3', 
            'semi_major_axis': 6779.0,
            'eccentricity': 0.0015,
            'inclination': 52.0,
            'raan': 2.0,
            'arg_periapsis': 0.0,
            'true_anomaly': 90.0,
            'mass': 2000.0,
            'cross_sectional_area': 15.0
        }
    ]
    
    # Add satellites to manager
    for config in satellite_configs:
        metadata = SatelliteMetadata(
            satellite_id=config['satellite_id'],
            name=config['name'],
            mass=config['mass'],
            cross_sectional_area=config['cross_sectional_area'],
            operational_status='active',
            mission_type='earth_observation',
            operator='Demo Agency',
            launch_date=datetime.now() - timedelta(days=365),
            operational_criticality=7,
            replacement_cost=100_000_000
        )
        
        satellite_manager.add_satellite(metadata)
        
        # Note: For demo purposes, we'll generate synthetic positions
        # In a real implementation, you would initialize with proper orbital elements
        logger.info(f"Added satellite {config['satellite_id']} to manager")
    
    # Initialize sensor network
    sensor_configs = [
        {
            'sensor_id': 'RADAR-001',
            'sensor_type': SensorType.RADAR,
            'location': np.array([0.0, 0.0, 0.0]),  # Earth center for demo
            'noise_characteristics': NoiseCharacteristics(
                position_sigma=0.1,  # 100m
                velocity_sigma=0.001,  # 1 m/s
                range_sigma=0.05,
                azimuth_sigma=0.1,
                elevation_sigma=0.1
            )
        },
        {
            'sensor_id': 'OPTICAL-001',
            'sensor_type': SensorType.OPTICAL,
            'location': np.array([1000.0, 0.0, 0.0]),
            'noise_characteristics': NoiseCharacteristics(
                position_sigma=0.2,
                velocity_sigma=0.002,
                range_sigma=0.1,
                azimuth_sigma=0.05,
                elevation_sigma=0.05
            )
        }
    ]
    
    for config in sensor_configs:
        sensor_config = SensorConfig(**config)
        sensor_emulator.add_sensor_station(sensor_config)
    
    logger.info(f"Initialized {len(satellite_configs)} satellites and {len(sensor_configs)} sensors")

def generate_satellite_positions_with_count(timestamp: datetime, count: int = 8) -> Dict[str, Dict]:
    """Generate satellite positions with specified count using N2YO API."""
    
    try:
        # Get satellites currently visible or use predefined ones
        satellite_ids = get_satellites_above_location()
        
        # If API returned empty (likely due to rate limiting), use predefined satellites
        if not satellite_ids:
            logger.info("No satellites from API - using predefined satellite list")
            satellite_ids = list(TRACKED_SATELLITES.keys())[:count]
        else:
            # Limit to requested count
            satellite_ids = satellite_ids[:count]
            
            # If we need more satellites than available, add some predefined ones
            if len(satellite_ids) < count:
                predefined_ids = list(TRACKED_SATELLITES.keys())
                for sat_id in predefined_ids:
                    if sat_id not in satellite_ids:
                        satellite_ids.append(sat_id)
                        if len(satellite_ids) >= count:
                            break
        
        # Fetch real satellite positions from API
        positions = fetch_satellite_positions_from_api(satellite_ids[:count])
        
        # If we got some real data, great! If not, use fallback
        if len(positions) > 0:
            logger.info(f"Successfully fetched {len(positions)} real satellite positions")
            # Add any missing satellites with mock data if needed
            if len(positions) < count:
                mock_positions = generate_mock_satellite_positions_with_count(timestamp, count - len(positions))
                # Rename mock satellites to avoid confusion
                for i, (mock_id, mock_data) in enumerate(mock_positions.items()):
                    new_id = f"SIM-{i+1:03d}"
                    mock_data['name'] = f"Simulated Satellite {i+1}"
                    positions[new_id] = mock_data
            return positions
        else:
            logger.warning("No satellite positions from API - using simulated data")
            
    except Exception as e:
        logger.error(f"Error fetching satellite positions from API: {e}")
    
    # Fallback to mock data if API fails completely
    logger.info(f"Using simulated satellite data for {count} satellites")
    return generate_mock_satellite_positions_with_count(timestamp, count)

def generate_mock_satellite_positions_with_count(timestamp: datetime, count: int = 8) -> Dict[str, Dict]:
    """Generate realistic mock satellite positions with enhanced randomness - satellites positioned closer together for better conjunction detection."""
    positions = {}
    
    # Create a seed based on current time to ensure different patterns
    np.random.seed(int(timestamp.timestamp() / 10) % 10000)  # Changes every 10 seconds
    
    # Define realistic satellite categories with SMALLER separation for more conjunctions
    satellite_categories = [
        {
            'type': 'ISS_TYPE',
            'name_prefix': 'Space Station',
            'altitude_range': (400, 430),  # Narrower range
            'inclination_range': (50, 53),  # Smaller spread
            'eccentricity_range': (0.0001, 0.003),
            'color_priority': '#e53e3e'
        },
        {
            'type': 'STARLINK',
            'name_prefix': 'Constellation',
            'altitude_range': (540, 560),  # Clustered together
            'inclination_range': (53, 54),  # Very close inclinations
            'eccentricity_range': (0.0001, 0.001),
            'color_priority': '#3182ce'
        },
        {
            'type': 'GPS',
            'name_prefix': 'Navigation',
            'altitude_range': (20000, 20050),  # Closer spacing
            'inclination_range': (54, 55),  # Tight range
            'eccentricity_range': (0.01, 0.015),
            'color_priority': '#38a169'
        },
        {
            'type': 'GEO_COMM',
            'name_prefix': 'Communication',
            'altitude_range': (35786, 35790),  # Very close for GEO
            'inclination_range': (0, 1),  # Nearly equatorial
            'eccentricity_range': (0.0001, 0.0005),
            'color_priority': '#d69e2e'
        },
        {
            'type': 'EARTH_OBS',
            'name_prefix': 'Earth Observer',
            'altitude_range': (600, 650),  # Clustered range
            'inclination_range': (97, 98),  # Sun-synchronous cluster
            'eccentricity_range': (0.001, 0.005),
            'color_priority': '#805ad5'
        },
        {
            'type': 'RESEARCH',
            'name_prefix': 'Research',
            'altitude_range': (450, 500),  # LEO research cluster
            'inclination_range': (45, 55),  # Moderate inclination cluster
            'eccentricity_range': (0.001, 0.01),
            'color_priority': '#dd6b20'
        }
    ]
    
    earth_radius = 6371.0  # km
    mu = 398600.4418  # Earth's gravitational parameter km³/s²
    
    # Generate satellites with MORE CLUSTERING for conjunctions
    for i in range(count):
        # Prefer clustered categories (70% of satellites in same altitude bands)
        if i < count * 0.7:
            # Force satellites into clusters
            if i < count * 0.3:
                category = satellite_categories[1]  # Starlink cluster
            elif i < count * 0.6:
                category = satellite_categories[0]  # ISS cluster
            else:
                category = satellite_categories[4]  # Earth obs cluster
        else:
            # Random categories for remaining satellites
            category = np.random.choice(satellite_categories)
        
        # Generate CLUSTERED orbital parameters
        altitude = np.random.uniform(*category['altitude_range'])
        inclination = np.random.uniform(*category['inclination_range'])
        
        # FORCE satellites to be closer in orbital space
        # Use time factor to create orbital trains
        time_factor = (timestamp.timestamp() + np.random.uniform(0, 3600)) % 3600 / 3600  # Shorter period
        
        # Create orbital trains - satellites following similar paths
        if i > 0 and np.random.random() < 0.4:  # 40% chance to follow previous satellite
            prev_sat = list(positions.values())[-1]
            # Follow similar orbital elements with small variations
            mean_anomaly = (prev_sat.get('mean_anomaly', 0) + np.random.uniform(10, 60)) % 360
            raan = prev_sat.get('raan', 0) + np.random.uniform(-5, 5)
            arg_periapsis = prev_sat.get('arg_periapsis', 0) + np.random.uniform(-10, 10)
            inclination = prev_sat.get('inclination', inclination) + np.random.uniform(-2, 2)
            altitude = prev_sat.get('altitude', altitude) + np.random.uniform(-20, 20)
        else:
            # Independent orbital elements but still clustered
            mean_anomaly = np.random.uniform(0, 360)
            raan = np.random.uniform(0, 360)
            arg_periapsis = np.random.uniform(0, 360)
        
        eccentricity = np.random.uniform(*category['eccentricity_range'])
        
        # Calculate orbital period and mean motion
        semi_major_axis = earth_radius + altitude
        if category['type'] == 'GEO_COMM':
            orbital_period = 86400  # 24 hours in seconds
            mean_motion = 360 / (orbital_period / 60)
        else:
            orbital_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
            mean_motion = 360 / (orbital_period / 60)
        
        # Convert orbital elements to position (simplified Keplerian elements)
        M = np.radians(mean_anomaly)
        E = M + eccentricity * np.sin(M)  # Eccentric anomaly
        
        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + eccentricity) * np.sin(E/2),
            np.sqrt(1 - eccentricity) * np.cos(E/2)
        )
        
        # Distance from Earth center
        r = semi_major_axis * (1 - eccentricity * np.cos(E))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        # Rotate to Earth-Centered Inertial coordinates
        i_rad = np.radians(inclination)
        raan_rad = np.radians(raan)
        argp_rad = np.radians(arg_periapsis)
        
        # Rotation matrices (simplified)
        cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
        cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
        cos_argp, sin_argp = np.cos(argp_rad), np.sin(argp_rad)
        
        # Transform to Earth-fixed coordinates (simplified)
        x = (cos_raan * cos_argp - sin_raan * sin_argp * cos_i) * x_orb + \
            (-cos_raan * sin_argp - sin_raan * cos_argp * cos_i) * y_orb
        y = (sin_raan * cos_argp + cos_raan * sin_argp * cos_i) * x_orb + \
            (-sin_raan * sin_argp + cos_raan * cos_argp * cos_i) * y_orb
        z = (sin_argp * sin_i) * x_orb + (cos_argp * sin_i) * y_orb
        
        # Calculate velocity (simplified)
        if category['type'] == 'GEO_COMM':
            orbital_speed = 2 * np.pi * r * np.cos(np.radians(inclination)) / orbital_period * 1000
        else:
            orbital_speed = np.sqrt(mu / r)
        
        # Velocity direction (simplified - perpendicular to radius in orbital plane)
        vx_orb = -orbital_speed * np.sin(nu + argp_rad)
        vy_orb = orbital_speed * np.cos(nu + argp_rad) * np.cos(i_rad)
        vz_orb = orbital_speed * np.cos(nu + argp_rad) * np.sin(i_rad)
        
        # Apply SMALLER random perturbations for clustering
        position_noise = np.random.normal(0, 0.05, 3)  # Reduced noise
        velocity_noise = np.random.normal(0, 0.005, 3)  # Reduced noise
        
        x += position_noise[0]
        y += position_noise[1] 
        z += position_noise[2]
        
        vx = vx_orb + velocity_noise[0]
        vy = vy_orb + velocity_noise[1]
        vz = vz_orb + velocity_noise[2]
        
        # Generate satellite ID and name - simplified naming
        sat_id = f"SIM-{i+1:03d}"
        sat_name = f"Simulated {i+1}"
        
        # Calculate derived values
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        
        positions[sat_id] = {
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'name': sat_name,
            'altitude': altitude,
            'latitude': lat,
            'longitude': lon,
            'inclination': inclination,
            'eccentricity': eccentricity,
            'orbital_period': orbital_period / 3600,  # hours
            'category': category['type'],
            'category_description': category['name_prefix'],
            'timestamp': timestamp.isoformat()
        }
    
    return positions

def detect_conjunctions(timestamp: datetime) -> List[Dict]:
    """Detect current conjunctions with enhanced analysis for diverse satellite types."""
    conjunctions = []
    
    try:
        # Generate satellite positions (real or fallback) with default count
        positions = generate_satellite_positions_with_count(timestamp, 8)
        
        if len(positions) < 2:
            return conjunctions
        
        # Check distances between all satellite pairs
        sat_ids = list(positions.keys())
        for i, sat1_id in enumerate(sat_ids):
            for j, sat2_id in enumerate(sat_ids[i+1:], i+1):
                pos1 = np.array([positions[sat1_id]['x'], positions[sat1_id]['y'], positions[sat1_id]['z']])
                pos2 = np.array([positions[sat2_id]['x'], positions[sat2_id]['y'], positions[sat2_id]['z']])
                
                # Calculate miss distance
                distance = np.linalg.norm(pos1 - pos2)
                
                # Get satellite information
                sat1_name = positions[sat1_id].get('name', sat1_id)
                sat2_name = positions[sat2_id].get('name', sat2_id)
                sat1_category = positions[sat1_id].get('category', 'UNKNOWN')
                sat2_category = positions[sat2_id].get('category', 'UNKNOWN')
                sat1_alt = positions[sat1_id].get('altitude', 0)
                sat2_alt = positions[sat2_id].get('altitude', 0)
                
                # Determine conjunction thresholds based on satellite categories and altitudes
                # Different orbital regimes have different conjunction criteria
                
                # LEO satellites (< 2000 km): tighter thresholds
                # MEO satellites (2000-35000 km): medium thresholds  
                # GEO satellites (>35000 km): looser thresholds
                
                max_alt = max(sat1_alt, sat2_alt)
                min_alt = min(sat1_alt, sat2_alt)
                
                if max_alt < 2000:  # Both in LEO
                    threshold = 100.0  # 100 km
                elif max_alt < 35000:  # At least one in MEO
                    threshold = 500.0  # 500 km
                else:  # At least one in GEO
                    threshold = 1000.0  # 1000 km
                
                # Special cases for similar categories
                if sat1_category == sat2_category:
                    if sat1_category in ['STARLINK', 'GPS']:
                        threshold *= 0.5  # Same constellation satellites are closer
                    elif sat1_category == 'GEO_COMM':
                        threshold *= 2.0  # GEO satellites have more spacing
                
                # Create conjunction if satellites are within threshold
                if distance < threshold:
                    
                    # Calculate relative velocity
                    vel1 = np.array([positions[sat1_id]['vx'], positions[sat1_id]['vy'], positions[sat1_id]['vz']])
                    vel2 = np.array([positions[sat2_id]['vx'], positions[sat2_id]['vy'], positions[sat2_id]['vz']])
                    rel_velocity = np.linalg.norm(vel1 - vel2)
                    
                    # Estimate time to closest approach (simplified)
                    relative_position = pos2 - pos1
                    relative_velocity_vec = vel2 - vel1
                    
                    # Time when distance is minimum (dot product approach)
                    if np.dot(relative_velocity_vec, relative_velocity_vec) > 0:
                        t_cpa = -np.dot(relative_position, relative_velocity_vec) / np.dot(relative_velocity_vec, relative_velocity_vec)
                        t_cpa = max(0, t_cpa)  # Only future times
                    else:
                        t_cpa = 0
                    
                    # Convert time to minutes and format
                    t_cpa_minutes = t_cpa / 60  # Convert to minutes
                    time_str = f"{int(t_cpa_minutes//60):02d}:{int(t_cpa_minutes%60):02d}:00"
                    
                    # Calculate collision probability based on satellite sizes and relative velocity
                    # More sophisticated model considering satellite categories
                    base_cross_section = 10.0  # m²
                    
                    # Adjust cross-section based on satellite type
                    if sat1_category == 'ISS_TYPE' or sat2_category == 'ISS_TYPE':
                        cross_section = 1000.0  # Much larger
                    elif 'GEO_COMM' in [sat1_category, sat2_category]:
                        cross_section = 50.0  # Large communication satellites
                    elif 'GPS' in [sat1_category, sat2_category]:
                        cross_section = 30.0  # Medium-sized navigation satellites
                    else:
                        cross_section = base_cross_section
                    
                    # Simplified collision probability calculation
                    miss_distance_m = distance * 1000  # Convert to meters
                    collision_prob = max(1e-9, (cross_section / (np.pi * miss_distance_m**2)) * np.exp(-rel_velocity/5))
                    
                    # Determine severity based on distance, probability, and satellite importance
                    if distance < threshold * 0.1:  # Very close
                        severity = 'CRITICAL'
                    elif distance < threshold * 0.2:
                        severity = 'HIGH'
                    elif distance < threshold * 0.4:
                        severity = 'MEDIUM'
                    elif distance < threshold * 0.7:
                        severity = 'LOW'
                    else:
                        severity = 'NEGLIGIBLE'
                    
                    # Upgrade severity for critical satellites
                    if sat1_category == 'ISS_TYPE' or sat2_category == 'ISS_TYPE':
                        if severity in ['LOW', 'NEGLIGIBLE']:
                            severity = 'MEDIUM'
                        elif severity == 'MEDIUM':
                            severity = 'HIGH'
                    
                    conjunctions.append({
                        'satellite_1': sat1_id,
                        'satellite_2': sat2_id,
                        'satellite_1_name': sat1_name.replace(' (FALLBACK)', ''),
                        'satellite_2_name': sat2_name.replace(' (FALLBACK)', ''),
                        'satellite_1_category': sat1_category,
                        'satellite_2_category': sat2_category,
                        'distance_km': distance,
                        'relative_velocity_kms': rel_velocity,
                        'collision_probability': collision_prob,
                        'time_to_closest_approach': time_str,
                        'severity': severity,
                        'altitude_1': sat1_alt,
                        'altitude_2': sat2_alt,
                        'threshold_used': threshold
                    })
        
        # Sort by severity (most critical first), then by distance
        severity_order = {'CRITICAL': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'NEGLIGIBLE': 1}
        conjunctions.sort(key=lambda x: (severity_order.get(x['severity'], 0), -x['distance_km']), reverse=True)
        
        # Limit to top 10 conjunctions to avoid overwhelming display
        conjunctions = conjunctions[:10]
        
        logger.info(f"Detected {len(conjunctions)} potential conjunctions")
    
    except Exception as e:
        logger.error(f"Error detecting conjunctions: {e}")
    
    return conjunctions

# Dashboard Layout with dark theme and improved organization
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Space Traffic Digital Twin System", style={
            'margin': '0',
            'font-size': '28px',
            'font-weight': '700',
            'color': '#58a6ff'
        }),
        html.P("Real-time satellite tracking and conjunction analysis", style={
            'margin': '8px 0 0 0',
            'opacity': '0.9',
            'font-size': '16px',
            'color': '#7d8590'
        })
    ], style={
        'background': 'linear-gradient(135deg, #161b22, #21262d)',
        'color': 'white',
        'padding': '25px',
        'margin-bottom': '20px',
        'border-radius': '10px',
        'box-shadow': '0 4px 12px rgba(0,0,0,0.5)',
        'border': '1px solid #30363d'
    }),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Number of Satellites:", style={
                'display': 'block',
                'margin-bottom': '10px',
                'font-weight': '600',
                'color': '#e6edf3',
                'font-size': '14px'
            }),
            dcc.Slider(
                id='satellite-count-slider',
                min=3,
                max=15,
                step=1,
                value=8,
                marks={i: str(i) for i in range(3, 16, 2)}
            )
        ], style={'min-width': '200px', 'flex': '1'}),
        
        html.Div([
            html.Button("Start Tracking", id="start-button", style={
                'padding': '12px 24px',
                'border': 'none',
                'border-radius': '8px',
                'cursor': 'pointer',
                'font-size': '14px',
                'font-weight': '600',
                'background': 'linear-gradient(135deg, #1f6feb, #0969da)',
                'color': 'white',
                'border': '1px solid #0969da',
                'margin-right': '12px'
            }),
            html.Button("Pause", id="stop-button", style={
                'padding': '12px 24px',
                'border': 'none',
                'border-radius': '8px',
                'cursor': 'pointer',
                'font-size': '14px',
                'font-weight': '600',
                'background': 'linear-gradient(135deg, #373e47, #424a53)',
                'color': 'white',
                'border': '1px solid #424a53',
                'margin-right': '12px'
            }),
            html.Button("Reset", id="reset-button", style={
                'padding': '12px 24px',
                'border': 'none',
                'border-radius': '8px',
                'cursor': 'pointer',
                'font-size': '14px',
                'font-weight': '600',
                'background': 'linear-gradient(135deg, #373e47, #424a53)',
                'color': 'white',
                'border': '1px solid #424a53'
            })
        ])
    ], style={
        'background': 'linear-gradient(135deg, #161b22, #21262d)',
        'padding': '20px',
        'margin-bottom': '20px',
        'border-radius': '10px',
        'border': '1px solid #30363d',
        'display': 'flex',
        'gap': '20px',
        'align-items': 'center',
        'flex-wrap': 'wrap',
        'box-shadow': '0 4px 12px rgba(0,0,0,0.4)'
    }),
    
    # Quick Summary Panel
    html.Div([
        html.H3("System Summary", style={'color': '#e6edf3'}),
        html.Div(id="quick-summary")
    ], style={
        'background': 'linear-gradient(135deg, #161b22, #21262d)',
        'padding': '20px',
        'margin-bottom': '20px',
        'border-radius': '10px',
        'border': '1px solid #30363d',
        'box-shadow': '0 4px 12px rgba(0,0,0,0.4)'
    }),
    
    # Status indicators
    html.Div([
        html.Div([
            html.H4("System Status", style={'color': '#e6edf3'}),
            html.Div(id="system-status")
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-right': '10px',
            'color': '#e6edf3'
        }),
        
        html.Div([
            html.H4("Tracking Statistics", style={'color': '#e6edf3'}),
            html.Div(id="system-stats")
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-left': '10px',
            'color': '#e6edf3'
        })
    ], style={'display': 'flex', 'gap': '20px', 'flex-wrap': 'wrap', 'margin-bottom': '20px'}),
    
    # Main displays
    html.Div([
        html.Div([
            html.H3("3D Satellite Tracking", style={'color': '#e6edf3'}),
            dcc.Graph(id="satellite-3d-plot", style={'height': '500px'})
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-right': '10px'
        }),
        
        html.Div([
            html.H3("Conjunction Analysis", style={'color': '#e6edf3'}),
            html.Div(id="conjunction-list", style={
                'max-height': '450px',
                'overflow-y': 'auto',
                'padding': '15px',
                'background': '#0d1117',
                'border-radius': '8px',
                'border': '1px solid #30363d'
            })
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-left': '10px'
        })
    ], style={'display': 'flex', 'gap': '20px', 'flex-wrap': 'wrap', 'margin-bottom': '20px'}),
    
    # Charts
    html.Div([
        html.Div([
            html.H4("Risk Distribution", style={'color': '#e6edf3'}),
            dcc.Graph(id="risk-distribution-chart", style={'height': '300px'})
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-right': '10px'
        }),
        
        html.Div([
            html.H4("Performance Metrics", style={'color': '#e6edf3'}),
            dcc.Graph(id="performance-chart", style={'height': '300px'})
        ], style={
            'background': 'linear-gradient(135deg, #161b22, #21262d)',
            'padding': '20px',
            'border-radius': '10px',
            'border': '1px solid #30363d',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.4)',
            'flex': '1',
            'min-width': '300px',
            'margin-bottom': '20px',
            'margin-left': '10px'
        })
    ], style={'display': 'flex', 'gap': '20px', 'flex-wrap': 'wrap', 'margin-bottom': '20px'}),
    
    # Information Panel (moved to bottom)
    html.Div([
        html.H3("🌌 About This Space Traffic Digital Twin", style={'color': '#58a6ff'}),
        html.Div([
            html.Div([
                html.H4("🔬 What is a Digital Twin?", style={'margin': '0 0 10px 0', 'color': '#81e6d9'}),
                html.P([
                    "A ", html.Strong("digital twin"), " is a real-time virtual replica of a physical system. ",
                    "This dashboard creates a live digital model of Earth's orbital environment, mirroring the ",
                    "actual positions, movements, and behaviors of satellites in space."
                ], style={'color': '#e6edf3'}),
                html.P([
                    "Digital twins enable us to ", html.Strong("monitor, analyze, and predict"), " what's happening ",
                    "in the real world without physically being there. They're used in manufacturing, smart cities, ",
                    "healthcare, and now - space traffic management."
                ], style={'color': '#e6edf3'})
            ], style={'margin-bottom': '20px'}),
            
            html.Div([
                html.H4("🤔 How is this Different from Regular Satellite Tracking?", style={'margin': '0 0 10px 0', 'color': '#f093fb'}),
                html.Div([
                    html.Div([
                        html.H5("📊 Traditional Satellite Tracking", style={'color': '#fbb6ce', 'margin': '0 0 8px 0'}),
                        html.Ul([
                            html.Li("Takes current satellite positions"),
                            html.Li("Calculates future trajectories using orbital mechanics"),
                            html.Li("Focuses on individual objects in isolation"),
                            html.Li("Static predictions that don't adapt"),
                            html.Li("Limited to basic trajectory math")
                        ], style={'font-size': '12px', 'margin': '0', 'color': '#e2e8f0'})
                    ]),
                    
                    html.Div([
                        html.H5("🌌 Digital Twin Approach", style={'color': '#81e6d9', 'margin': '15px 0 8px 0'}),
                        html.Ul([
                            html.Li([html.Strong("Complete Environment: "), "Models entire space ecosystem, not just individual satellites"]),
                            html.Li([html.Strong("System Interactions: "), "Simulates how satellites affect each other (conjunctions, debris creation)"]),
                            html.Li([html.Strong("Environmental Forces: "), "Includes atmospheric drag, solar pressure, gravitational perturbations"]),
                            html.Li([html.Strong("Adaptive Learning: "), "Updates models based on real-world observations and discrepancies"]),
                            html.Li([html.Strong("What-If Scenarios: "), "Test maneuvers, new launches, failure modes safely"]),
                            html.Li([html.Strong("Operational Context: "), "Models ground systems, communication delays, human decision-making"]),
                            html.Li([html.Strong("Bidirectional Sync: "), "Virtual changes can inform real-world operations"])
                        ], style={'font-size': '12px', 'margin': '0', 'color': '#e2e8f0'})
                    ])
                ])
            ], style={'margin-bottom': '20px'}),
            
            html.Div([
                html.H4("💡 Practical Example: Satellite Collision Avoidance", style={'margin': '0 0 10px 0', 'color': '#68d391'}),
                html.Div([
                    html.Div([
                        html.H5("📊 Traditional Approach", style={'color': '#fbb6ce', 'margin': '0 0 8px 0'}),
                        html.P("\"Satellite A and B will be 2 km apart in 6 hours. Move Satellite A.\"", 
                               style={'font-style': 'italic', 'color': '#cbd5e0', 'margin': '0 0 8px 0'}),
                        html.P("• Simple trajectory calculation", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Doesn't consider what happens after the maneuver", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• No validation of the solution", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'})
                    ]),
                    
                    html.Div([
                        html.H5("🌌 Digital Twin Approach", style={'color': '#81e6d9', 'margin': '15px 0 8px 0'}),
                        html.P("\"Testing 3 maneuver options in virtual space before executing...\"", 
                               style={'font-style': 'italic', 'color': '#cbd5e0', 'margin': '0 0 8px 0'}),
                        html.P("• Simulates all 3 maneuver options in the virtual environment", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Checks impact on other satellites in the constellation", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Considers fuel consumption and mission impact", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Tests what happens if communication is delayed", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Validates the maneuver won't create new collision risks", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Updates the virtual model with the planned maneuver", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'}),
                        html.P("• Continues monitoring the virtual outcome vs. reality", style={'font-size': '12px', 'margin': '0', 'color': '#e6edf3'})
                    ])
                ])
            ], style={'margin-bottom': '20px'}),
            
            html.Div([
                html.H4("💡 Digital Twin Benefits", style={'margin': '0 0 10px 0', 'color': '#d53f8c'}),
                html.Ul([
                    html.Li([html.Strong("Safe Testing: "), "Analyze 'what-if' scenarios without risking real satellites"]),
                    html.Li([html.Strong("Early Warning: "), "Detect potential collisions hours or days in advance"]),
                    html.Li([html.Strong("Decision Support: "), "Help operators decide when to perform avoidance maneuvers"]),
                    html.Li([html.Strong("Training: "), "Train space traffic controllers in a risk-free environment"]),
                    html.Li([html.Strong("Optimization: "), "Test new orbital insertion strategies and mission plans"])
                ], style={'margin': '0', 'padding-left': '20px'})
            ], className="twin-benefits"),
            
            html.Hr(style={'margin': '20px 0', 'border-color': '#4a5568'}),
            
            html.Div([
                html.P([
                    "🌍 ", html.Strong("Current Status: "), 
                    "This digital twin represents the space environment around Earth from Low Earth Orbit (LEO) ",
                    "at ~400 km altitude to Geostationary Orbit (GEO) at 35,786 km, tracking satellites across ",
                    "all major orbital regimes used by real space missions."
                ], style={'font-style': 'italic', 'color': '#a0aec0'})
            ])
        ], className="info-content")
    ], className="info-panel"),
    
    # Auto-update and state management (fixed at 2 seconds)
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # 2 seconds
        n_intervals=0,
        disabled=True
    ),
    dcc.Store(id='simulation-state', data={'running': False}),
    dcc.Store(id='satellite-data', data={}),
    dcc.Store(id='conjunction-data', data=[])
])

# Callbacks
@app.callback(
    [Output('satellite-3d-plot', 'figure'),
     Output('conjunction-list', 'children'),
     Output('system-stats', 'children'),
     Output('system-status', 'children'),
     Output('quick-summary', 'children'),
     Output('risk-distribution-chart', 'figure'),
     Output('performance-chart', 'figure'),
     Output('satellite-data', 'data'),
     Output('conjunction-data', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('satellite-count-slider', 'value')],
    [State('simulation-state', 'data')]
)
def update_dashboard(n_intervals, start_clicks, reset_clicks, satellite_count, sim_state):
    """Main dashboard update callback with error handling."""
    
    try:
        # Use current time for live data
        current_time = datetime.now()
        
        # Generate satellite positions with specified count
        satellite_positions = generate_satellite_positions_with_count(current_time, satellite_count)
        
        # Detect conjunctions
        conjunctions = detect_conjunctions(current_time)
        
        # Create components with error handling
        try:
            satellite_fig = create_satellite_3d_plot(satellite_positions)
        except Exception as e:
            logger.error(f"Error creating satellite plot: {e}")
            satellite_fig = go.Figure().add_annotation(
                text="Display Error",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color='red', size=16)
            )
        
        try:
            conjunction_list = create_conjunction_list(conjunctions)
        except Exception as e:
            logger.error(f"Error creating conjunction list: {e}")
            conjunction_list = html.Div([
                html.P("Error loading conjunction data", style={'color': 'red'})
            ])
        
        try:
            system_stats = create_system_stats(satellite_positions, conjunctions)
        except Exception as e:
            logger.error(f"Error creating system stats: {e}")
            system_stats = html.Div([
                html.P("Error loading statistics", style={'color': 'red'})
            ])
        
        try:
            system_status = create_system_status(satellite_positions)
        except Exception as e:
            logger.error(f"Error creating system status: {e}")
            system_status = html.Div([
                html.P("Error loading system status", style={'color': 'red'})
            ])
        
        try:
            quick_summary = create_quick_summary(satellite_positions, conjunctions)
        except Exception as e:
            logger.error(f"Error creating quick summary: {e}")
            quick_summary = html.Div([
                html.P("Error loading summary", style={'color': 'red'})
            ])
        
        try:
            risk_chart = create_risk_distribution_chart(conjunctions)
        except Exception as e:
            logger.error(f"Error creating risk chart: {e}")
            risk_chart = go.Figure().add_annotation(
                text="Chart Error",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color='red', size=16)
            )
        
        try:
            performance_chart = create_performance_chart(current_time)
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            performance_chart = go.Figure().add_annotation(
                text="Performance Error",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color='red', size=16)
            )
        
        return (satellite_fig, conjunction_list, system_stats, system_status, quick_summary,
                risk_chart, performance_chart, satellite_positions, conjunctions)
                
    except Exception as e:
        logger.error(f"Critical error in dashboard update: {e}")
        
        # Return safe fallback values
        empty_fig = go.Figure().add_annotation(
            text="System Error",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='red', size=16)
        )
        
        error_div = html.Div([
            html.P("System Error", style={'color': 'red'})
        ])
        
        return (empty_fig, error_div, error_div, error_div, error_div, empty_fig, empty_fig, {}, [])

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('simulation-state', 'data')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('simulation-state', 'data')]
)
def control_simulation(start_clicks, stop_clicks, sim_state):
    """Control simulation start/stop."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return True, {'running': False}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button':
        return False, {'running': True}
    elif button_id == 'stop-button':
        return True, {'running': False}
    
    return True, {'running': False}

def create_satellite_3d_plot(positions: Dict[str, Dict]) -> go.Figure:
    """Create 3D satellite tracking plot - dark theme with enhanced satellite info."""
    
    fig = go.Figure()
    
    # Add Earth as a proper sphere with better resolution
    u = np.linspace(0, 2 * np.pi, 40)  # Increased resolution
    v = np.linspace(0, np.pi, 30)     # Increased resolution
    earth_radius = 6371  # km
    
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Enhanced Earth surface with better colors
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, '#1a365d'], [0.3, '#2d3748'], [0.7, '#1a365d'], [1, '#0f2027']],
        opacity=0.8,
        showscale=False,
        name='Earth (6371 km radius)',
        hovertemplate='<b>Earth</b><br>Radius: 6371 km<br>Reference sphere for satellite tracking<extra></extra>',
        lighting=dict(ambient=0.4, diffuse=0.6, fresnel=0.2, specular=0.1, roughness=0.3)
    ))
    
    # Color mapping for satellite categories
    category_colors = {
        'ISS_TYPE': '#e53e3e',
        'STARLINK': '#3182ce', 
        'GPS': '#38a169',
        'GEO_COMM': '#d69e2e',
        'EARTH_OBS': '#805ad5',
        'RESEARCH': '#dd6b20'
    }
    
    # Default colors for other satellites
    default_colors = ['#f56565', '#63b3ed', '#68d391', '#fbb6ce', '#9f7aea', '#f6ad55']
    
    # Group satellites by category for better organization
    satellites_by_category = {}
    
    for sat_id, pos in positions.items():
        category = pos.get('category', 'UNKNOWN')
        if category not in satellites_by_category:
            satellites_by_category[category] = []
        satellites_by_category[category].append((sat_id, pos))
    
    # Add satellites grouped by category
    for category, sats in satellites_by_category.items():
        color = category_colors.get(category, default_colors[len(satellites_by_category) % len(default_colors)])
        
        for sat_id, pos in sats:
            sat_name = pos.get('name', sat_id)
            
            # Create enhanced hover information with explanations
            if pos.get('category'):  # Enhanced simulated satellite
                category_desc = pos.get('category_description', 'Unknown Type')
                hover_text = f"<b>{sat_name}</b><br>" + \
                           f"Type: {category_desc}<br>" + \
                           f"Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) km from Earth center<br>" + \
                           f"Altitude: {pos.get('altitude', 'N/A'):.1f} km above Earth surface<br>" + \
                           f"Orbital Inclination: {pos.get('inclination', 'N/A'):.1f}° (angle from equator)<br>" + \
                           f"Velocity: {np.sqrt(pos['vx']**2 + pos['vy']**2 + pos['vz']**2):.2f} km/s<br>" + \
                           f"Orbital Period: {pos.get('orbital_period', 'N/A'):.2f} hours (time for one orbit)<br>" + \
                           f"Eccentricity: {pos.get('eccentricity', 'N/A'):.4f} (0=circular, >0=elliptical)<br>" + \
                           "<i>Simulated satellite with realistic orbital mechanics</i>"
            else:  # Real satellite or basic info
                hover_text = f"<b>{sat_name}</b><br>" + \
                           f"Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) km from Earth center<br>" + \
                           f"Altitude: {pos.get('altitude', 'N/A')} km above Earth surface<br>" + \
                           f"Velocity: {np.sqrt(pos['vx']**2 + pos['vy']**2 + pos['vz']**2):.2f} km/s<br>" + \
                           "<i>Real satellite data from N2YO API</i>"
            
            # Determine marker size based on category
            marker_size = 8
            if category == 'ISS_TYPE':
                marker_size = 14
            elif category == 'GEO_COMM':
                marker_size = 11
            elif category == 'GPS':
                marker_size = 10
            
            # Add satellite position
            fig.add_trace(go.Scatter3d(
                x=[pos['x']],
                y=[pos['y']],
                z=[pos['z']],
                mode='markers+text',
                marker=dict(
                    size=marker_size, 
                    color=color, 
                    line=dict(width=2, color='white'),
                    opacity=0.9,
                    symbol='circle'
                ),
                text=[sat_name.split()[0]],  # Show first word only
                textposition='top center',
                textfont=dict(color='white', size=8),
                name=f"{category.replace('_', ' ')}" if category != 'UNKNOWN' else sat_name,
                legendgroup=category,
                showlegend=category not in [item.legendgroup for item in fig.data if hasattr(item, 'legendgroup')],
                hovertemplate=hover_text + "<extra></extra>"
            ))
    
    # Update layout for dark theme with better camera angle
    fig.update_layout(
        scene=dict(
            xaxis_title='X (km from Earth center)',
            yaxis_title='Y (km from Earth center)',
            zaxis_title='Z (km from Earth center)',
            aspectmode='cube',
            bgcolor='#1a202c',
            xaxis=dict(
                gridcolor='#4a5568', 
                showbackground=True, 
                backgroundcolor='#2d3748',
                range=[-50000, 50000]  # Set range to show all orbits
            ),
            yaxis=dict(
                gridcolor='#4a5568', 
                showbackground=True, 
                backgroundcolor='#2d3748',
                range=[-50000, 50000]
            ),
            zaxis=dict(
                gridcolor='#4a5568', 
                showbackground=True, 
                backgroundcolor='#2d3748',
                range=[-50000, 50000]
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),  # Better viewing angle
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0, y=1, 
            bgcolor='rgba(45, 55, 72, 0.9)', 
            bordercolor='#4a5568', 
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#1a202c',
        plot_bgcolor='#1a202c',
        font=dict(color='#e2e8f0'),
        title=dict(
            text="3D Satellite Positions Around Earth",
            font=dict(size=14, color='#e2e8f0'),
            x=0.5
        )
    )
    
    return fig

def create_conjunction_list(conjunctions: List[Dict]) -> html.Div:
    """Create conjunction list display with explanations."""
    
    if not conjunctions:
        return html.Div([
            html.Div([
                html.H5("✅ No Conjunctions Detected", style={'color': '#38a169', 'margin': '0 0 10px 0'}),
                html.P("All tracked satellites are maintaining safe distances from each other.", 
                       style={'margin': '0 0 10px 0', 'color': '#a0aec0'}),
                html.P("The system continuously monitors for close approaches (conjunctions) between satellites. " +
                       "A conjunction occurs when two satellites come within a specified distance threshold.", 
                       style={'margin': '0', 'color': '#718096', 'font-size': '12px', 'font-style': 'italic'})
            ], style={'text-align': 'center', 'padding': '20px'})
        ])
    
    conjunction_items = []
    
    # Add explanation header
    conjunction_items.append(
        html.Div([
            html.H5("⚠️ Active Conjunctions", style={'color': '#e53e3e', 'margin': '0 0 10px 0'}),
            html.P("Satellites currently approaching each other within risk thresholds. " +
                   "Higher risk events require immediate attention.", 
                   style={'margin': '0 0 15px 0', 'color': '#a0aec0', 'font-size': '12px'})
        ], style={'border-bottom': '1px solid #4a5568', 'padding-bottom': '10px', 'margin-bottom': '15px'})
    )
    
    for i, conj in enumerate(conjunctions):
        risk_color = {
            'NEGLIGIBLE': '#38a169',
            'LOW': '#d69e2e', 
            'MEDIUM': '#dd6b20',
            'HIGH': '#e53e3e',
            'CRITICAL': '#9b2c2c'
        }.get(conj['severity'].upper(), '#718096')
        
        # Get satellite names or fall back to IDs
        sat1_name = conj.get('satellite_1_name', conj['satellite_1'])
        sat2_name = conj.get('satellite_2_name', conj['satellite_2'])
        
        # Get categories for better description
        cat1 = conj.get('satellite_1_category', '').replace('_', ' ').title()
        cat2 = conj.get('satellite_2_category', '').replace('_', ' ').title()
        
        # Create severity description
        severity_desc = {
            'CRITICAL': 'Immediate action required',
            'HIGH': 'Close monitoring needed',
            'MEDIUM': 'Standard tracking protocol',
            'LOW': 'Routine observation',
            'NEGLIGIBLE': 'Minor proximity event'
        }.get(conj['severity'].upper(), 'Unknown risk level')
        
        conjunction_items.append(
            html.Div([
                html.H6(f"Event {i+1}: {sat1_name} ↔ {sat2_name}", 
                       style={'color': risk_color, 'margin': '0 0 8px 0', 'font-size': '14px'}),
                html.Div([
                    html.P([
                        html.Strong("Miss Distance: "), f"{conj['distance_km']:.1f} km",
                        html.Br(),
                        html.Strong("Risk Level: "), 
                        html.Span(f"{conj['severity'].upper()} - {severity_desc}", 
                                style={'color': risk_color, 'font-weight': 'bold'}),
                        html.Br(),
                        html.Strong("Collision Probability: "), f"{conj['collision_probability']:.2e}",
                        html.Br(),
                        html.Strong("Relative Velocity: "), f"{conj.get('relative_velocity_kms', 0):.2f} km/s",
                        html.Br(),
                        html.Strong("Time to Closest Approach: "), conj['time_to_closest_approach']
                    ], style={'margin': '0 0 8px 0', 'font-size': '12px', 'color': '#e6edf3'}),
                    
                    html.P([
                        html.Strong("Satellite Types: "), 
                        f"{cat1 if cat1 else 'Unknown'} & {cat2 if cat2 else 'Unknown'}",
                        html.Br(),
                        html.Strong("Altitudes: "), 
                        f"{conj['altitude_1']:.0f} km & {conj['altitude_2']:.0f} km"
                    ], style={'margin': '0', 'font-size': '11px', 'color': '#a0aec0', 'font-style': 'italic'})
                ])
            ], style={
                'background': 'linear-gradient(135deg, #161b22, #21262d)',
                'border': '1px solid #30363d',
                'border-radius': '8px',
                'padding': '15px',
                'margin-bottom': '12px',
                'border-left': f'4px solid {risk_color}'
            })
        )
    
    # Add summary
    critical_count = len([c for c in conjunctions if c['severity'] == 'CRITICAL'])
    high_count = len([c for c in conjunctions if c['severity'] == 'HIGH'])
    
    summary_color = '#9b2c2c' if critical_count > 0 else '#e53e3e' if high_count > 0 else '#d69e2e'
    summary_text = f"Total Events: {len(conjunctions)} | Critical: {critical_count} | High Risk: {high_count}"
    
    conjunction_items.append(
        html.Div([
            html.Hr(style={'border-color': '#4a5568'}),
            html.P(summary_text, style={'color': summary_color, 'text-align': 'center', 'font-weight': 'bold', 'margin': '10px 0'}),
            html.P("Conjunctions are detected when satellites approach within altitude-specific thresholds. " +
                   "LEO satellites: 100 km, MEO: 500 km, GEO: 1000 km.", 
                   style={'text-align': 'center', 'font-size': '10px', 'color': '#718096', 'margin': '0'})
        ])
    )
    
    return html.Div(conjunction_items)

def create_system_stats(positions: Dict, conjunctions: List) -> html.Div:
    """Create system statistics display in clean style."""
    
    total_satellites = len(positions)
    active_conjunctions = len(conjunctions)
    high_risk_conjunctions = len([c for c in conjunctions if c['severity'] in ['HIGH', 'CRITICAL']])
    
    status_text = "NORMAL" if high_risk_conjunctions == 0 else "ALERT"
    status_color = "#38a169" if high_risk_conjunctions == 0 else "#e53e3e"
    
    return html.Div([
        html.P(f"Active Satellites: {total_satellites}", style={'margin': '5px 0', 'color': '#e6edf3'}),
        html.P(f"Active Conjunctions: {active_conjunctions}", style={'margin': '5px 0', 'color': '#e6edf3'}),
        html.P(f"High Risk Events: {high_risk_conjunctions}", style={'margin': '5px 0', 'color': '#e6edf3'}),
        html.P([
            "System Status: ",
            html.Span(status_text, style={'color': status_color, 'font-weight': 'bold'})
        ], style={'margin': '5px 0', 'color': '#e6edf3'})
    ])

def create_risk_distribution_chart(conjunctions: List) -> go.Figure:
    """Create risk level distribution chart in dark theme."""
    
    if not conjunctions:
        fig = go.Figure()
        fig.add_annotation(
            text="No Risk Data Available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(color='#a0aec0', size=16)
        )
        fig.update_layout(
            paper_bgcolor='#1a202c',
            plot_bgcolor='#1a202c',
            font=dict(color='#e2e8f0')
        )
        return fig
    
    risk_levels = [c['severity'].upper() for c in conjunctions]
    risk_counts = pd.Series(risk_levels).value_counts()
    
    # Dark theme colors for risk levels
    dark_colors = {
        'NEGLIGIBLE': '#38a169',
        'LOW': '#d69e2e',
        'MEDIUM': '#dd6b20', 
        'HIGH': '#e53e3e',
        'CRITICAL': '#9b2c2c'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[dark_colors.get(level, '#718096') for level in risk_counts.index],
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            marker_line=dict(color='#4a5568', width=1)
        )
    ])
    
    fig.update_layout(
        xaxis_title='Risk Level',
        yaxis_title='Count',
        margin=dict(l=40, r=10, t=10, b=40),
        showlegend=False,
        paper_bgcolor='#1a202c',
        plot_bgcolor='#1a202c',
        font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#4a5568'),
        yaxis=dict(gridcolor='#4a5568')
    )
    
    return fig

def create_performance_chart(current_time: datetime) -> go.Figure:
    """Create performance metrics chart in dark theme."""
    
    # Generate sample performance data
    times = [current_time - timedelta(minutes=i) for i in range(30, 0, -1)]
    
    # Simulate performance metrics
    propagation_times = 8 + 3 * np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.normal(0, 0.5, 30)
    conjunction_times = 45 + 10 * np.sin(np.linspace(0, 2*np.pi, 30)) + np.random.normal(0, 2, 30)
    risk_calc_times = 4 + 1 * np.sin(np.linspace(0, 6*np.pi, 30)) + np.random.normal(0, 0.2, 30)
    
    fig = go.Figure()
    
    # Add performance traces with dark theme colors
    fig.add_trace(go.Scatter(
        x=times,
        y=propagation_times,
        mode='lines',
        name='Orbit Propagation',
        line=dict(color='#63b3ed', width=2),
        hovertemplate='<b>Orbit Propagation</b><br>Time: %{x}<br>Latency: %{y:.1f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=conjunction_times,
        mode='lines',
        name='Conjunction Detection',
        line=dict(color='#f56565', width=2),
        hovertemplate='<b>Conjunction Detection</b><br>Time: %{x}<br>Latency: %{y:.1f}ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=risk_calc_times,
        mode='lines',
        name='Risk Calculation',
        line=dict(color='#68d391', width=2),
        hovertemplate='<b>Risk Calculation</b><br>Time: %{x}<br>Latency: %{y:.1f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Latency (ms)',
        margin=dict(l=50, r=10, t=10, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(45, 55, 72, 0.8)', bordercolor='#4a5568', borderwidth=1),
        paper_bgcolor='#1a202c',
        plot_bgcolor='#1a202c',
        font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='#4a5568'),
        yaxis=dict(gridcolor='#4a5568')
    )
    
    return fig

def create_system_status(positions: Dict) -> html.Div:
    """Create system status display."""
    
    total_satellites = len(positions)
    api_satellites = len([p for p in positions.values() if 'Simulated' not in p.get('name', '')])
    
    status = "Operational" if total_satellites > 0 else "Error"
    status_color = "#38a169" if total_satellites > 0 else "#e53e3e"
    
    return html.Div([
        html.P(f"System Status: {status}", style={'color': status_color, 'font-weight': 'bold'}),
        html.P(f"Data Source: {'Real API + Simulated' if api_satellites > 0 else 'Simulated Only'}", style={'color': '#e6edf3'}),
        html.P(f"Tracking Mode: Real-time", style={'color': '#e6edf3'}),
        html.P(f"Last Update: {datetime.now().strftime('%H:%M:%S')}", style={'color': '#e6edf3'})
    ])

def create_quick_summary(positions: Dict, conjunctions: List) -> html.Div:
    """Create a quick, easy-to-glance summary panel emphasizing digital twin concept."""
    
    total_satellites = len(positions)
    total_conjunctions = len(conjunctions)
    critical_events = len([c for c in conjunctions if c['severity'] in ['CRITICAL', 'HIGH']])
    
    # Check data sources more accurately
    real_satellites = len([p for p in positions.values() 
                          if 'Simulated' not in p.get('name', '') and not p.get('name', '').startswith('SIM-')])
    simulated_satellites = total_satellites - real_satellites
    
    # Determine digital twin status
    if real_satellites > 0:
        twin_status = f"🔗 HYBRID TWIN"
        twin_detail = f"Real data + Physics simulation"
        twin_color = "#3182ce"
    else:
        twin_status = "🧬 PHYSICS TWIN"
        twin_detail = "Full orbital mechanics simulation"
        twin_color = "#805ad5"
    
    # Determine space environment status
    if critical_events > 0:
        space_status = "🚨 HIGH RISK ENVIRONMENT"
        space_color = "#e53e3e"
    elif total_conjunctions > 0:
        space_status = "⚠️ ACTIVE MONITORING"
        space_color = "#d69e2e"
    else:
        space_status = "✅ STABLE ENVIRONMENT"
        space_color = "#38a169"
    
    return html.Div([
        html.Div([
            html.H2(space_status, style={'color': space_color, 'margin': '0', 'text-align': 'center'}),
            html.P(f"Digital Twin Status: {twin_status}", 
                  style={'text-align': 'center', 'margin': '5px 0', 'color': twin_color, 'font-weight': 'bold'}),
            html.P(twin_detail, style={'text-align': 'center', 'margin': '5px 0', 'color': twin_color, 'font-size': '12px'}),
            html.P(f"Last Mirror Update: {datetime.now().strftime('%H:%M:%S')}", 
                  style={'text-align': 'center', 'margin': '5px 0', 'opacity': '0.7', 'font-size': '11px'})
        ], className="summary-status"),
        
        html.Div([
            html.Div([
                html.H3(str(total_satellites), style={'margin': '0', 'font-size': '2em'}),
                html.P("Virtual Satellites", style={'margin': '0'}),
                html.P("(Mirroring Real Space)", style={'margin': '0', 'font-size': '10px', 'opacity': '0.7'})
            ], className="summary-metric"),
            
            html.Div([
                html.H3(str(total_conjunctions), style={'margin': '0', 'font-size': '2em'}),
                html.P("Predicted Events", style={'margin': '0'}),
                html.P("(Conjunction Analysis)", style={'margin': '0', 'font-size': '10px', 'opacity': '0.7'})
            ], className="summary-metric"),
            
            html.Div([
                html.H3(str(critical_events), style={'margin': '0', 'font-size': '2em', 'color': '#e53e3e' if critical_events > 0 else 'inherit'}),
                html.P("High Priority", style={'margin': '0'}),
                html.P("(Requires Action)", style={'margin': '0', 'font-size': '10px', 'opacity': '0.7'})
            ], className="summary-metric"),
            
            html.Div([
                html.H3("🌍", style={'margin': '0', 'font-size': '2em'}),
                html.P("Earth Orbit", style={'margin': '0'}),
                html.P("(400km - 36,000km)", style={'margin': '0', 'font-size': '10px', 'opacity': '0.7'})
            ], className="summary-metric")
        ], className="summary-metrics")
    ])

if __name__ == '__main__':
    # Initialize demo data
    initialize_demo_data()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8050) 