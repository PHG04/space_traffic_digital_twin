"""
Risk Assessment and Calculation Engine

Provides comprehensive risk analysis including collision probability assessment,
risk metrics calculation, and multi-dimensional risk evaluation.
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Handle both relative and absolute imports
try:
    from ..conjunction_detection.conjunction_analyzer import ConjunctionEvent, ConjunctionSeverity
    from ..data_management.satellite_manager import SatelliteMetadata
except ImportError:
    # If relative import fails, try absolute import for when running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from conjunction_detection.conjunction_analyzer import ConjunctionEvent, ConjunctionSeverity
    from data_management.satellite_manager import SatelliteMetadata


class RiskLevel(Enum):
    """Risk level classifications."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a satellite or conjunction."""
    
    collision_probability: float
    risk_level: RiskLevel
    economic_risk_usd: float
    mission_criticality_impact: float
    debris_generation_potential: float
    cascade_risk_factor: float
    maneuver_urgency_score: float
    time_to_closest_approach_hours: float
    confidence_level: float
    
    # Contributing factors
    uncertainty_factor: float
    tracking_quality: float
    conjunction_frequency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'collision_probability': self.collision_probability,
            'risk_level': self.risk_level.value,
            'economic_risk_usd': self.economic_risk_usd,
            'mission_criticality_impact': self.mission_criticality_impact,
            'debris_generation_potential': self.debris_generation_potential,
            'cascade_risk_factor': self.cascade_risk_factor,
            'maneuver_urgency_score': self.maneuver_urgency_score,
            'time_to_closest_approach_hours': self.time_to_closest_approach_hours,
            'confidence_level': self.confidence_level,
            'uncertainty_factor': self.uncertainty_factor,
            'tracking_quality': self.tracking_quality,
            'conjunction_frequency': self.conjunction_frequency
        }


class RiskCalculator:
    """
    Advanced risk assessment calculator for satellite operations.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize risk calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.pc_thresholds = {
            RiskLevel.NEGLIGIBLE: 1e-9,
            RiskLevel.LOW: 1e-6,
            RiskLevel.MEDIUM: 1e-5,
            RiskLevel.HIGH: 1e-4,
            RiskLevel.CRITICAL: 1e-3
        }
        
        # Economic factors
        self.default_satellite_value = self.config.get('default_satellite_value_usd', 100_000_000)
        self.debris_cleanup_cost = self.config.get('debris_cleanup_cost_usd', 50_000_000)
        self.mission_disruption_cost_per_day = self.config.get('mission_disruption_cost_per_day', 1_000_000)
        
        # Risk weighting factors
        self.economic_weight = self.config.get('economic_weight', 0.3)
        self.safety_weight = self.config.get('safety_weight', 0.4)
        self.mission_weight = self.config.get('mission_weight', 0.3)
        
        # Time constants
        self.urgent_maneuver_threshold_hours = self.config.get('urgent_maneuver_threshold_hours', 24)
        self.normal_maneuver_threshold_hours = self.config.get('normal_maneuver_threshold_hours', 72)
        
        self.logger.info("Risk Calculator initialized")
    
    def calculate_conjunction_risk(self, 
                                 conjunction: ConjunctionEvent,
                                 satellite1_meta: Optional[SatelliteMetadata] = None,
                                 satellite2_meta: Optional[SatelliteMetadata] = None,
                                 historical_conjunctions: List[ConjunctionEvent] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a conjunction event.
        
        Args:
            conjunction: Conjunction event to assess
            satellite1_meta: Metadata for first satellite
            satellite2_meta: Metadata for second satellite
            historical_conjunctions: Historical conjunction data for frequency analysis
            
        Returns:
            Comprehensive risk metrics
        """
        try:
            # Basic probability and timing
            pc = conjunction.collision_probability
            time_to_ca = (conjunction.time_of_closest_approach - datetime.now()).total_seconds() / 3600.0
            
            # Calculate economic risk
            economic_risk = self._calculate_economic_risk(pc, satellite1_meta, satellite2_meta)
            
            # Calculate mission criticality impact
            mission_impact = self._calculate_mission_impact(satellite1_meta, satellite2_meta)
            
            # Calculate debris generation potential
            debris_potential = self._calculate_debris_potential(
                conjunction.closest_approach_distance,
                conjunction.relative_velocity,
                satellite1_meta,
                satellite2_meta
            )
            
            # Calculate cascade risk
            cascade_risk = self._calculate_cascade_risk(conjunction, satellite1_meta, satellite2_meta)
            
            # Calculate maneuver urgency
            urgency_score = self._calculate_maneuver_urgency(pc, time_to_ca, conjunction.severity)
            
            # Calculate uncertainty and quality factors
            uncertainty_factor = self._calculate_uncertainty_factor(conjunction)
            tracking_quality = self._calculate_tracking_quality(conjunction)
            
            # Calculate conjunction frequency
            conjunction_freq = self._calculate_conjunction_frequency(
                conjunction, historical_conjunctions or []
            )
            
            # Determine overall risk level
            risk_level = self._determine_risk_level(pc, urgency_score, mission_impact)
            
            # Calculate confidence level
            confidence = self._calculate_confidence_level(
                tracking_quality, uncertainty_factor, time_to_ca
            )
            
            return RiskMetrics(
                collision_probability=pc,
                risk_level=risk_level,
                economic_risk_usd=economic_risk,
                mission_criticality_impact=mission_impact,
                debris_generation_potential=debris_potential,
                cascade_risk_factor=cascade_risk,
                maneuver_urgency_score=urgency_score,
                time_to_closest_approach_hours=time_to_ca,
                confidence_level=confidence,
                uncertainty_factor=uncertainty_factor,
                tracking_quality=tracking_quality,
                conjunction_frequency=conjunction_freq
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating conjunction risk: {e}")
            # Return default low-confidence metrics
            return self._get_default_risk_metrics()
    
    def _calculate_economic_risk(self, 
                               pc: float,
                               satellite1_meta: Optional[SatelliteMetadata],
                               satellite2_meta: Optional[SatelliteMetadata]) -> float:
        """Calculate economic risk in USD."""
        
        # Satellite values
        sat1_value = satellite1_meta.replacement_cost if satellite1_meta else self.default_satellite_value
        sat2_value = satellite2_meta.replacement_cost if satellite2_meta else self.default_satellite_value
        
        # Expected loss from satellite destruction
        satellite_loss_risk = pc * (sat1_value + sat2_value)
        
        # Debris cleanup costs
        debris_risk = pc * self.debris_cleanup_cost
        
        # Mission disruption costs (estimated)
        disruption_days = max(30, min(365, 1/max(pc, 1e-9)))  # Higher Pc = shorter disruption estimate
        mission_disruption_risk = pc * self.mission_disruption_cost_per_day * disruption_days
        
        total_economic_risk = satellite_loss_risk + debris_risk + mission_disruption_risk
        
        return total_economic_risk
    
    def _calculate_mission_impact(self,
                                satellite1_meta: Optional[SatelliteMetadata],
                                satellite2_meta: Optional[SatelliteMetadata]) -> float:
        """Calculate mission criticality impact (0-1 scale)."""
        
        # Default criticality if no metadata
        default_criticality = 5
        
        criticality1 = satellite1_meta.operational_criticality if satellite1_meta else default_criticality
        criticality2 = satellite2_meta.operational_criticality if satellite2_meta else default_criticality
        
        # Normalize to 0-1 scale and take maximum impact
        normalized_impact = max(criticality1, criticality2) / 10.0
        
        return normalized_impact
    
    def _calculate_debris_potential(self,
                                  miss_distance: float,
                                  relative_velocity: float,
                                  satellite1_meta: Optional[SatelliteMetadata],
                                  satellite2_meta: Optional[SatelliteMetadata]) -> float:
        """Calculate potential for debris generation (0-1 scale)."""
        
        # Default masses if no metadata
        default_mass = 1000.0  # kg
        
        mass1 = satellite1_meta.mass if satellite1_meta and hasattr(satellite1_meta, 'mass') else default_mass
        mass2 = satellite2_meta.mass if satellite2_meta and hasattr(satellite2_meta, 'mass') else default_mass
        
        # Kinetic energy factor
        total_mass = mass1 + mass2
        kinetic_energy = 0.5 * total_mass * (relative_velocity ** 2)
        
        # Distance factor (closer = more debris)
        distance_factor = max(0.0, 1.0 - miss_distance / 10.0)  # Significant debris risk within 10 km
        
        # Energy factor (higher energy = more debris)
        energy_factor = min(1.0, kinetic_energy / 1e12)  # Normalize by typical collision energy
        
        debris_potential = distance_factor * energy_factor
        
        return min(1.0, max(0.0, debris_potential))
    
    def _calculate_cascade_risk(self,
                              conjunction: ConjunctionEvent,
                              satellite1_meta: Optional[SatelliteMetadata],
                              satellite2_meta: Optional[SatelliteMetadata]) -> float:
        """Calculate cascade/Kessler syndrome risk factor (0-1 scale)."""
        
        # Get orbital altitudes (approximate from conjunction data)
        # In practice, would use actual orbital elements
        estimated_altitude = 400.0  # Default LEO altitude
        
        # Altitude risk factor (LEO has higher cascade risk)
        if estimated_altitude < 600:  # LEO
            altitude_risk = 1.0
        elif estimated_altitude < 1500:  # MEO
            altitude_risk = 0.5
        else:  # GEO
            altitude_risk = 0.2
        
        # Debris generation potential contributes to cascade risk
        debris_potential = self._calculate_debris_potential(
            conjunction.closest_approach_distance,
            conjunction.relative_velocity,
            satellite1_meta,
            satellite2_meta
        )
        
        cascade_risk = altitude_risk * debris_potential
        
        return min(1.0, max(0.0, cascade_risk))
    
    def _calculate_maneuver_urgency(self,
                                  pc: float,
                                  time_to_ca_hours: float,
                                  severity: ConjunctionSeverity) -> float:
        """Calculate maneuver urgency score (0-1 scale)."""
        
        # Probability urgency
        pc_urgency = min(1.0, pc / 1e-4)  # Max urgency at Pc = 1e-4
        
        # Time urgency
        if time_to_ca_hours <= 0:
            time_urgency = 1.0
        elif time_to_ca_hours <= self.urgent_maneuver_threshold_hours:
            time_urgency = 1.0 - (time_to_ca_hours / self.urgent_maneuver_threshold_hours) * 0.5
        elif time_to_ca_hours <= self.normal_maneuver_threshold_hours:
            time_urgency = 0.5 - (time_to_ca_hours - self.urgent_maneuver_threshold_hours) / \
                          (self.normal_maneuver_threshold_hours - self.urgent_maneuver_threshold_hours) * 0.4
        else:
            time_urgency = 0.1  # Minimum urgency for distant events
        
        # Severity multiplier
        severity_multipliers = {
            ConjunctionSeverity.CRITICAL: 1.0,
            ConjunctionSeverity.HIGH: 0.8,
            ConjunctionSeverity.MEDIUM: 0.6,
            ConjunctionSeverity.LOW: 0.4
        }
        severity_multiplier = severity_multipliers.get(severity, 0.5)
        
        urgency_score = (pc_urgency + time_urgency) / 2.0 * severity_multiplier
        
        return min(1.0, max(0.0, urgency_score))
    
    def _calculate_uncertainty_factor(self, conjunction: ConjunctionEvent) -> float:
        """Calculate uncertainty factor based on conjunction data (0-1 scale)."""
        
        # Use dilution threshold as uncertainty measure
        uncertainty = min(1.0, conjunction.dilution_threshold / 10.0)  # Normalize by 10 km
        
        return uncertainty
    
    def _calculate_tracking_quality(self, conjunction: ConjunctionEvent) -> float:
        """Calculate tracking quality factor (0-1 scale)."""
        
        # Simplified quality based on miss distance accuracy
        # Lower miss distances with higher certainty indicate better tracking
        miss_distance = conjunction.closest_approach_distance
        
        if miss_distance < 1.0:  # Very close - need high accuracy
            quality = 1.0 - min(0.5, conjunction.dilution_threshold / 1.0)
        else:  # Further apart - accuracy less critical
            quality = 0.8
        
        return max(0.1, min(1.0, quality))
    
    def _calculate_conjunction_frequency(self,
                                       conjunction: ConjunctionEvent,
                                       historical_conjunctions: List[ConjunctionEvent]) -> float:
        """Calculate conjunction frequency for the satellite pair."""
        
        # Count historical conjunctions between the same satellite pair
        satellite_pair = {conjunction.satellite_1, conjunction.satellite_2}
        
        relevant_conjunctions = [
            c for c in historical_conjunctions
            if {c.satellite_1, c.satellite_2} == satellite_pair
        ]
        
        # Calculate frequency over past 30 days
        cutoff_time = datetime.now() - timedelta(days=30)
        recent_conjunctions = [
            c for c in relevant_conjunctions
            if c.detection_time >= cutoff_time
        ]
        
        frequency = len(recent_conjunctions) / 30.0  # Conjunctions per day
        
        return min(10.0, frequency)  # Cap at 10 per day
    
    def _determine_risk_level(self,
                            pc: float,
                            urgency_score: float,
                            mission_impact: float) -> RiskLevel:
        """Determine overall risk level based on multiple factors."""
        
        # Primary classification by Pc
        if pc >= self.pc_thresholds[RiskLevel.CRITICAL]:
            base_level = RiskLevel.CRITICAL
        elif pc >= self.pc_thresholds[RiskLevel.HIGH]:
            base_level = RiskLevel.HIGH
        elif pc >= self.pc_thresholds[RiskLevel.MEDIUM]:
            base_level = RiskLevel.MEDIUM
        elif pc >= self.pc_thresholds[RiskLevel.LOW]:
            base_level = RiskLevel.LOW
        else:
            base_level = RiskLevel.NEGLIGIBLE
        
        # Escalate based on urgency and mission criticality
        composite_factor = (urgency_score + mission_impact) / 2.0
        
        if composite_factor > 0.8 and base_level.value in ['low', 'negligible']:
            return RiskLevel.MEDIUM
        elif composite_factor > 0.9 and base_level == RiskLevel.MEDIUM:
            return RiskLevel.HIGH
        
        return base_level
    
    def _calculate_confidence_level(self,
                                  tracking_quality: float,
                                  uncertainty_factor: float,
                                  time_to_ca_hours: float) -> float:
        """Calculate confidence level in risk assessment (0-1 scale)."""
        
        # Base confidence from tracking quality
        base_confidence = tracking_quality
        
        # Reduce confidence with uncertainty
        uncertainty_penalty = uncertainty_factor * 0.3
        
        # Reduce confidence for distant events (propagation uncertainty)
        if time_to_ca_hours > 72:
            time_penalty = min(0.3, (time_to_ca_hours - 72) / 168 * 0.3)  # Max penalty at 1 week
        else:
            time_penalty = 0.0
        
        confidence = base_confidence - uncertainty_penalty - time_penalty
        
        return max(0.1, min(1.0, confidence))
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics for error cases."""
        return RiskMetrics(
            collision_probability=0.0,
            risk_level=RiskLevel.NEGLIGIBLE,
            economic_risk_usd=0.0,
            mission_criticality_impact=0.0,
            debris_generation_potential=0.0,
            cascade_risk_factor=0.0,
            maneuver_urgency_score=0.0,
            time_to_closest_approach_hours=1000.0,
            confidence_level=0.1,
            uncertainty_factor=1.0,
            tracking_quality=0.1,
            conjunction_frequency=0.0
        )
    
    def calculate_fleet_risk_summary(self, conjunction_risks: List[RiskMetrics]) -> Dict[str, Any]:
        """Calculate fleet-wide risk summary from individual conjunction risks."""
        
        if not conjunction_risks:
            return {
                'total_conjunctions': 0,
                'risk_distribution': {level.value: 0 for level in RiskLevel},
                'total_economic_risk_usd': 0.0,
                'average_collision_probability': 0.0,
                'urgent_maneuvers_needed': 0,
                'high_confidence_assessments': 0
            }
        
        # Risk level distribution
        risk_distribution = {level.value: 0 for level in RiskLevel}
        for risk in conjunction_risks:
            risk_distribution[risk.risk_level.value] += 1
        
        # Economic risk
        total_economic_risk = sum(risk.economic_risk_usd for risk in conjunction_risks)
        
        # Average Pc
        avg_pc = np.mean([risk.collision_probability for risk in conjunction_risks])
        
        # Urgent maneuvers
        urgent_maneuvers = sum(1 for risk in conjunction_risks if risk.maneuver_urgency_score > 0.7)
        
        # High confidence assessments
        high_confidence = sum(1 for risk in conjunction_risks if risk.confidence_level > 0.8)
        
        return {
            'total_conjunctions': len(conjunction_risks),
            'risk_distribution': risk_distribution,
            'total_economic_risk_usd': total_economic_risk,
            'average_collision_probability': avg_pc,
            'urgent_maneuvers_needed': urgent_maneuvers,
            'high_confidence_assessments': high_confidence,
            'confidence_rate': high_confidence / len(conjunction_risks)
        } 