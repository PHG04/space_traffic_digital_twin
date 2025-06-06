# Space Traffic Management (STM) Digital Twin

A comprehensive real-time space traffic management system for monitoring satellite conjunctions, collision risk assessment, and space situational awareness.

## 🚀 Features

### Core Capabilities
- **Real-time Orbital Propagation**: High-performance Keplerian and J2 perturbation models supporting 1000+ satellites at 1 Hz
- **Conjunction Detection**: KDTree-based spatial indexing for efficient proximity detection (<5km threshold)
- **Collision Probability Calculation**: Sophisticated algorithms including Chan, Foster, and Alfano methods
- **Risk Assessment**: Multi-dimensional risk analysis with economic impact, mission criticality, and debris generation potential
- **Sensor Data Emulation**: Realistic noise modeling for radar, optical, and other tracking systems
- **3D Visualization Dashboard**: Interactive web-based monitoring and control interface

### Technical Highlights
- **Scalable Architecture**: Modular design supporting distributed processing
- **High Performance**: Optimized algorithms with parallel processing and state caching
- **Realistic Modeling**: Physics-based orbital mechanics with perturbation effects
- **Comprehensive Risk Analysis**: Economic, safety, and mission impact assessment
- **Real-time Processing**: Sub-second response times for critical conjunction alerts

## 📋 System Requirements

### Dependencies
```
Python 3.8+
numpy
scipy
poliastro
astropy
plotly
dash
pandas
```

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended for 1000+ satellites)
- **RAM**: 16GB+ for large satellite fleets
- **Storage**: SSD recommended for optimal I/O performance

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd "Space Traffic Digital Twin"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -m pytest tests/ -v
```

### 4. Run Dashboard
```bash
cd dashboard
python app.py
```

Access the dashboard at: `http://localhost:8050`

## 🏗️ Architecture

### Module Structure
```
src/
├── orbital_mechanics/          # Orbit propagation and state management
│   ├── orbit_engine.py        # Main orchestrator
│   ├── propagators.py          # Keplerian and J2 propagators
│   ├── satellite_state.py     # State vector management
│   └── coordinate_transforms.py # Coordinate system conversions
├── data_management/            # Satellite fleet management
│   ├── satellite_manager.py   # Satellite metadata and lifecycle
│   └── real_time_propagation.py # High-performance propagation engine
├── spatial_indexing/           # Efficient proximity searches
│   └── kdtree_index.py        # KDTree-based spatial indexing
├── conjunction_detection/      # Conjunction analysis
│   └── conjunction_analyzer.py # Event detection and tracking
├── risk_assessment/            # Risk analysis and metrics
│   ├── risk_calculator.py     # Multi-dimensional risk assessment
│   └── collision_probability.py # Sophisticated Pc calculations
├── sensor_simulation/          # Realistic sensor modeling
│   ├── sensor_emulator.py     # Multi-sensor network simulation
│   └── noise_models.py        # Gaussian, radar, and correlated noise
└── utils/                      # Shared utilities
    ├── performance_monitor.py  # System performance tracking
    └── config_manager.py       # Configuration management
```

## 🚀 Quick Start

### Basic Usage Example
```python
from src.orbital_mechanics.orbit_engine import OrbitEngine
from src.data_management.satellite_manager import SatelliteManager
from src.conjunction_detection.conjunction_analyzer import ConjunctionAnalyzer
from datetime import datetime

# Initialize components
orbit_engine = OrbitEngine()
satellite_manager = SatelliteManager()
conjunction_analyzer = ConjunctionAnalyzer()

# Add satellites
satellite_manager.add_satellite_from_tle("SAT-001", tle_line1, tle_line2)

# Initialize orbit
orbit_engine.initialize_satellite_orbit(
    satellite_id="SAT-001",
    semi_major_axis=6778.0,  # km
    eccentricity=0.001,
    inclination=51.6,        # degrees
    raan=0.0,
    arg_periapsis=0.0,
    true_anomaly=0.0
)

# Propagate to current time
current_time = datetime.now()
state = orbit_engine.propagate_satellite("SAT-001", current_time)

print(f"Position: {state.state_vector.position} km")
print(f"Velocity: {state.state_vector.velocity} km/s")
```

### Conjunction Detection Example
```python
# Get all satellite states
satellite_states = {}
for sat_id in satellite_manager.get_all_satellite_ids():
    satellite_states[sat_id] = orbit_engine.propagate_satellite(sat_id, current_time)

# Detect conjunctions
conjunctions = conjunction_analyzer.detect_conjunctions(satellite_states, current_time)

for conjunction in conjunctions:
    print(f"Conjunction: {conjunction.satellite_1} ↔ {conjunction.satellite_2}")
    print(f"Miss Distance: {conjunction.closest_approach_distance:.3f} km")
    print(f"Collision Probability: {conjunction.collision_probability:.2e}")
```

### Risk Assessment Example
```python
from src.risk_assessment.risk_calculator import RiskCalculator

risk_calculator = RiskCalculator()

for conjunction in conjunctions:
    # Get satellite metadata
    sat1_meta = satellite_manager.get_satellite_metadata(conjunction.satellite_1)
    sat2_meta = satellite_manager.get_satellite_metadata(conjunction.satellite_2)
    
    # Calculate comprehensive risk metrics
    risk_metrics = risk_calculator.calculate_conjunction_risk(
        conjunction, sat1_meta, sat2_meta
    )
    
    print(f"Risk Level: {risk_metrics.risk_level.value}")
    print(f"Economic Risk: ${risk_metrics.economic_risk_usd:,.0f}")
    print(f"Maneuver Urgency: {risk_metrics.maneuver_urgency_score:.2f}")
```

## 📊 Dashboard Features

### Real-time Monitoring
- **3D Satellite Tracking**: Interactive visualization of satellite positions around Earth
- **Conjunction List**: Real-time display of active conjunctions with risk levels
- **System Statistics**: Fleet status, conjunction counts, and performance metrics
- **Risk Distribution**: Visual breakdown of risk levels across all conjunctions

### Control Interface
- **Simulation Controls**: Start/stop/reset simulation with configurable time steps
- **Time Navigation**: Jump to specific dates and times for analysis
- **Update Intervals**: Adjustable refresh rates from 1-60 seconds

### Performance Monitoring
- **Execution Times**: Real-time tracking of orbit propagation, conjunction detection, and risk calculation performance
- **Sensor Network Status**: Monitoring of sensor availability and measurement quality
- **System Health**: Overall system performance and resource utilization

## ⚙️ Configuration

### System Configuration (`config/system_config.yaml`)
```yaml
orbit_engine:
  propagation_method: "j2"
  time_step_seconds: 60
  max_satellites: 1000
  
conjunction_detection:
  proximity_threshold_km: 5.0
  time_window_hours: 72
  
risk_assessment:
  default_satellite_value_usd: 100000000
  debris_cleanup_cost_usd: 50000000
  
performance:
  enable_caching: true
  cache_size_mb: 512
  parallel_processing: true
```

### Sensor Configuration
```python
from src.sensor_simulation.sensor_emulator import SensorConfig, SensorType
from src.sensor_simulation.noise_models import NoiseCharacteristics

# Configure radar sensor
radar_config = SensorConfig(
    sensor_id="RADAR-001",
    sensor_type=SensorType.RADAR,
    location=np.array([0.0, 0.0, 0.0]),
    noise_characteristics=NoiseCharacteristics(
        position_sigma=0.1,      # 100m position uncertainty
        velocity_sigma=0.001,    # 1 m/s velocity uncertainty
        range_sigma=0.05,        # 50m range uncertainty
        azimuth_sigma=0.1,       # 0.1° azimuth uncertainty
        elevation_sigma=0.1      # 0.1° elevation uncertainty
    ),
    operational_range_km=3000.0,
    min_elevation_deg=10.0,
    availability=0.95
)
```

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_orbital_mechanics.py -v
python -m pytest tests/test_conjunction_detection.py -v
python -m pytest tests/test_risk_assessment.py -v

# Run performance tests
python -m pytest tests/test_performance.py -v --benchmark
```

### Performance Benchmarks
- **Orbit Propagation**: <10ms per satellite per time step
- **Conjunction Detection**: <50ms for 1000 satellites
- **Risk Calculation**: <5ms per conjunction
- **Overall System**: 1 Hz update rate for 1000+ satellites

## 📈 Performance Optimization

### High-Performance Configuration
```python
# Enable parallel processing
orbit_engine = OrbitEngine(config={
    'enable_parallel_processing': True,
    'num_workers': 8,
    'batch_size': 100
})

# Optimize spatial indexing
conjunction_analyzer = ConjunctionAnalyzer(config={
    'kdtree_leaf_size': 10,
    'enable_caching': True,
    'cache_ttl_seconds': 300
})

# Configure performance monitoring
from src.utils.performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor(enable_detailed_logging=True)
```

### Memory Management
```python
# Configure state caching
real_time_engine = RealTimePropagationEngine(config={
    'state_cache_size': 10000,
    'enable_drift_compensation': True,
    'cleanup_interval_seconds': 3600
})
```

## 🔧 Advanced Features

### Custom Propagators
```python
from src.orbital_mechanics.propagators import CustomPropagator

class MyPropagator(CustomPropagator):
    def propagate(self, state, dt):
        # Custom propagation logic
        return new_state

# Register custom propagator
orbit_engine.register_propagator("custom", MyPropagator())
```

### Risk Assessment Customization
```python
# Custom risk thresholds
risk_calculator = RiskCalculator(config={
    'pc_thresholds': {
        'negligible': 1e-9,
        'low': 1e-6,
        'medium': 1e-5,
        'high': 1e-4,
        'critical': 1e-3
    },
    'economic_weight': 0.3,
    'safety_weight': 0.4,
    'mission_weight': 0.3
})
```

### Sensor Network Customization
```python
# Multi-sensor fusion
sensor_emulator = SensorEmulator(config={
    'enable_sensor_fusion': True,
    'fusion_time_window_s': 5.0,
    'measurement_history_limit': 1000
})
```

## 📚 API Reference

### Core Classes

#### OrbitEngine
- `initialize_satellite_orbit()`: Initialize satellite orbital elements
- `propagate_satellite()`: Propagate satellite to specified time
- `propagate_all_satellites()`: Batch propagation for multiple satellites
- `get_performance_metrics()`: Retrieve propagation performance statistics

#### ConjunctionAnalyzer
- `detect_conjunctions()`: Detect conjunctions for satellite fleet
- `analyze_conjunction()`: Detailed analysis of specific conjunction
- `get_conjunction_history()`: Retrieve historical conjunction data

#### RiskCalculator
- `calculate_conjunction_risk()`: Comprehensive risk assessment
- `calculate_fleet_risk_summary()`: Fleet-wide risk analysis
- `update_risk_thresholds()`: Modify risk classification thresholds

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
flake8 src/
black src/
mypy src/
```

### Code Standards
- **Style**: Black code formatting
- **Type Hints**: Full type annotation required
- **Documentation**: Comprehensive docstrings
- **Testing**: >90% code coverage required

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Poliastro**: Astrodynamics library for Python
- **Astropy**: Astronomy and astrophysics library
- **SciPy**: Scientific computing library
- **Plotly/Dash**: Interactive visualization framework

## 📞 Support

For questions, issues, or contributions:
- **Issues**: GitHub Issues tracker
- **Documentation**: [Wiki](wiki-url)
- **Discussions**: [GitHub Discussions](discussions-url)

---

**Space Traffic Management Digital Twin** - Advancing space safety through real-time monitoring and risk assessment. 