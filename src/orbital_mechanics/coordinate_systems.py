"""
Coordinate System Transformations

Utilities for converting between different coordinate systems used in orbital mechanics.
"""

import numpy as np
from typing import Tuple
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import ITRS, GCRS, CartesianRepresentation


class CoordinateTransformer:
    """
    Coordinate system transformation utilities.
    """
    
    def __init__(self):
        """Initialize coordinate transformer."""
        # Earth rotation rate (rad/s)
        self.omega_earth = 7.2921159e-5
        
        # Earth constants
        self.earth_radius = 6378.137  # km
        self.earth_flattening = 1/298.257223563
    
    def eci_to_ecef(self, position_eci: np.ndarray, velocity_eci: np.ndarray, 
                    timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert from Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF).
        
        Args:
            position_eci: Position vector in ECI frame [km]
            velocity_eci: Velocity vector in ECI frame [km/s]
            timestamp: Time of observation
            
        Returns:
            Tuple of (position_ecef, velocity_ecef)
        """
        # Calculate Greenwich Mean Sidereal Time
        gmst = self._calculate_gmst(timestamp)
        
        # Rotation matrix from ECI to ECEF
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R = np.array([
            [cos_gmst, sin_gmst, 0],
            [-sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Rotation rate matrix
        omega_matrix = np.array([
            [0, -self.omega_earth, 0],
            [self.omega_earth, 0, 0],
            [0, 0, 0]
        ])
        
        # Transform position
        position_ecef = R @ position_eci
        
        # Transform velocity (includes Earth rotation effect)
        velocity_ecef = R @ velocity_eci - omega_matrix @ position_ecef
        
        return position_ecef, velocity_ecef
    
    def ecef_to_eci(self, position_ecef: np.ndarray, velocity_ecef: np.ndarray,
                    timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert from ECEF to ECI coordinate system.
        
        Args:
            position_ecef: Position vector in ECEF frame [km]
            velocity_ecef: Velocity vector in ECEF frame [km/s]
            timestamp: Time of observation
            
        Returns:
            Tuple of (position_eci, velocity_eci)
        """
        # Calculate Greenwich Mean Sidereal Time
        gmst = self._calculate_gmst(timestamp)
        
        # Rotation matrix from ECEF to ECI (transpose of ECI to ECEF)
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R = np.array([
            [cos_gmst, -sin_gmst, 0],
            [sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Rotation rate matrix
        omega_matrix = np.array([
            [0, -self.omega_earth, 0],
            [self.omega_earth, 0, 0],
            [0, 0, 0]
        ])
        
        # Transform position
        position_eci = R @ position_ecef
        
        # Transform velocity (includes Earth rotation effect)
        velocity_eci = R @ (velocity_ecef + omega_matrix @ position_ecef)
        
        return position_eci, velocity_eci
    
    def ecef_to_geodetic(self, position_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to geodetic coordinates (latitude, longitude, altitude).
        
        Args:
            position_ecef: Position vector in ECEF frame [km]
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_km)
        """
        x, y, z = position_ecef
        
        # Calculate longitude
        longitude = np.arctan2(y, x)
        
        # Calculate latitude and altitude using iterative method
        a = self.earth_radius  # Semi-major axis
        f = self.earth_flattening  # Flattening
        e2 = f * (2 - f)  # First eccentricity squared
        
        # Initial guess
        p = np.sqrt(x**2 + y**2)
        latitude = np.arctan2(z, p * (1 - e2))
        
        # Iterate to find accurate latitude and altitude
        for _ in range(10):
            N = a / np.sqrt(1 - e2 * np.sin(latitude)**2)
            altitude = p / np.cos(latitude) - N
            latitude_new = np.arctan2(z, p * (1 - e2 * N / (N + altitude)))
            
            if abs(latitude_new - latitude) < 1e-12:
                break
            latitude = latitude_new
        
        # Final altitude calculation
        N = a / np.sqrt(1 - e2 * np.sin(latitude)**2)
        altitude = p / np.cos(latitude) - N
        
        return np.degrees(latitude), np.degrees(longitude), altitude
    
    def geodetic_to_ecef(self, latitude_deg: float, longitude_deg: float, 
                        altitude_km: float) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF coordinates.
        
        Args:
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            altitude_km: Altitude above ellipsoid in km
            
        Returns:
            Position vector in ECEF frame [km]
        """
        lat = np.radians(latitude_deg)
        lon = np.radians(longitude_deg)
        alt = altitude_km
        
        a = self.earth_radius  # Semi-major axis
        f = self.earth_flattening  # Flattening
        e2 = f * (2 - f)  # First eccentricity squared
        
        # Radius of curvature in prime vertical
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        
        # ECEF coordinates
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt) * np.sin(lat)
        
        return np.array([x, y, z])
    
    def _calculate_gmst(self, timestamp: datetime) -> float:
        """
        Calculate Greenwich Mean Sidereal Time.
        
        Args:
            timestamp: UTC timestamp
            
        Returns:
            GMST in radians
        """
        # Convert to Julian date
        t = Time(timestamp)
        jd = t.jd
        
        # Days since J2000.0
        t_ut1 = (jd - 2451545.0)
        
        # GMST at 0h UT1 (IAU 2000 formula)
        gmst_0h = (24110.54841 + 8640184.812866 * t_ut1 / 36525.0 + 
                  0.093104 * (t_ut1 / 36525.0)**2 - 
                  6.2e-6 * (t_ut1 / 36525.0)**3)
        
        # Add the fraction of the day
        fraction_of_day = (jd - int(jd) - 0.5) % 1.0
        gmst_0h += fraction_of_day * 86400.0 * 1.00273790935
        
        # Convert to radians and normalize
        gmst = (gmst_0h % 86400.0) * (2 * np.pi / 86400.0)
        
        return gmst
    
    def calculate_ground_track(self, position_eci: np.ndarray, 
                             timestamp: datetime) -> Tuple[float, float]:
        """
        Calculate ground track (latitude, longitude) for a given ECI position.
        
        Args:
            position_eci: Position vector in ECI frame [km]
            timestamp: Time of observation
            
        Returns:
            Tuple of (latitude_deg, longitude_deg)
        """
        # Convert ECI to ECEF
        # For ground track, we only need position (velocity not required)
        dummy_velocity = np.zeros(3)
        position_ecef, _ = self.eci_to_ecef(position_eci, dummy_velocity, timestamp)
        
        # Convert ECEF to geodetic
        lat, lon, alt = self.ecef_to_geodetic(position_ecef)
        
        return lat, lon
    
    def calculate_look_angles(self, satellite_position: np.ndarray, 
                            observer_position: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate look angles (azimuth, elevation, range) from observer to satellite.
        
        Args:
            satellite_position: Satellite position in ECEF [km]
            observer_position: Observer position in ECEF [km]
            
        Returns:
            Tuple of (azimuth_deg, elevation_deg, range_km)
        """
        # Vector from observer to satellite
        range_vector = satellite_position - observer_position
        range_km = np.linalg.norm(range_vector)
        
        # Convert observer position to geodetic
        obs_lat, obs_lon, obs_alt = self.ecef_to_geodetic(observer_position)
        obs_lat_rad = np.radians(obs_lat)
        obs_lon_rad = np.radians(obs_lon)
        
        # Transformation matrix from ECEF to local ENU (East-North-Up)
        sin_lat = np.sin(obs_lat_rad)
        cos_lat = np.cos(obs_lat_rad)
        sin_lon = np.sin(obs_lon_rad)
        cos_lon = np.cos(obs_lon_rad)
        
        T = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        
        # Transform range vector to local ENU frame
        enu_vector = T @ range_vector
        east, north, up = enu_vector
        
        # Calculate azimuth and elevation
        azimuth = np.arctan2(east, north)
        if azimuth < 0:
            azimuth += 2 * np.pi
        
        elevation = np.arctan2(up, np.sqrt(east**2 + north**2))
        
        return np.degrees(azimuth), np.degrees(elevation), range_km 