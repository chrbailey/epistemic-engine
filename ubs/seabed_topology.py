"""
Seabed Topology Integration
===========================

Hidden variable that explains ocean anomalies humans struggle to grasp:

- Underwater canyons funnel cold water upward → temperature drops
- Seamounts create upwelling → nutrients → fish
- Depth gradients affect wave behavior
- Substrate type determines what lives there

KEY INSIGHT:
------------
When buoy 46042 (Monterey) shows 13.5°C water temp and buoy 46026 (SF)
shows 13.8°C, that's NOT an anomaly - it's explained by:
  - Monterey sits at edge of 2100m canyon (cold upwelling)
  - SF sits in 54m shallow water (warmer)

TruthLayer should LEARN these spatial explanations automatically.

DATA SOURCE:
------------
GEBCO 2020 via Open Topo Data API (free, JSON)
- 15 arc-second resolution (~450m)
- Global bathymetry/topography
"""

import json
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

from truth_layer import TruthLayer


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SeabedPoint:
    """Single point of seabed data."""
    lat: float
    lon: float
    depth_m: float  # Negative = below sea level

    @property
    def is_deep(self) -> bool:
        """Canyon/deep water (>500m)."""
        return self.depth_m < -500

    @property
    def is_shallow(self) -> bool:
        """Continental shelf (<200m)."""
        return -200 < self.depth_m < 0

    @property
    def depth_category(self) -> str:
        """Human-readable depth category."""
        d = self.depth_m
        if d >= 0:
            return "land"
        elif d > -50:
            return "very_shallow"
        elif d > -200:
            return "shallow_shelf"
        elif d > -500:
            return "mid_depth"
        elif d > -1000:
            return "deep"
        elif d > -2000:
            return "very_deep"
        else:
            return "abyssal"


@dataclass
class SeabedGrid:
    """Grid of seabed points for a region."""
    center_lat: float
    center_lon: float
    resolution_km: float
    points: List[SeabedPoint] = field(default_factory=list)

    # Computed features
    min_depth: float = 0
    max_depth: float = 0
    mean_depth: float = 0
    depth_variance: float = 0
    gradient_magnitude: float = 0  # How steep is the terrain
    canyon_nearby: bool = False
    upwelling_likely: bool = False


@dataclass
class SpatialBelief:
    """Belief about a spatial region that explains observations."""
    region_id: str
    belief_type: str  # canyon_effect, upwelling_zone, temperature_gradient
    description: str
    confidence: float
    evidence_count: int
    affected_metrics: List[str]  # water_temp, wave_height, etc.


# =============================================================================
# BATHYMETRY CLIENT
# =============================================================================

class BathymetryClient:
    """
    Fetch ocean depth data from Open Topo Data API.

    Uses GEBCO 2020 dataset:
    - 15 arc-second resolution (~450m at equator)
    - Global coverage
    - Free, no API key
    - 100 locations per request
    """

    BASE_URL = "https://api.opentopodata.org/v1/gebco2020"

    def get_depth(self, lat: float, lon: float) -> Optional[float]:
        """Get depth at a single point."""
        url = f"{self.BASE_URL}?locations={lat},{lon}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())

                if data.get("status") == "OK" and data.get("results"):
                    return data["results"][0].get("elevation")
        except Exception as e:
            print(f"  [Bathymetry] Error: {e}")

        return None

    def get_depths_batch(self, locations: List[Tuple[float, float]]) -> List[SeabedPoint]:
        """
        Get depths for multiple points (up to 100).

        Returns list of SeabedPoint objects.
        """
        if not locations:
            return []

        # Limit to 100 per API requirement
        locations = locations[:100]

        # Format: lat,lon|lat,lon|...
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in locations)
        url = f"{self.BASE_URL}?locations={loc_str}"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())

                if data.get("status") != "OK":
                    return []

                points = []
                for r in data.get("results", []):
                    if r.get("elevation") is not None:
                        points.append(SeabedPoint(
                            lat=r["location"]["lat"],
                            lon=r["location"]["lng"],
                            depth_m=r["elevation"]
                        ))
                return points

        except Exception as e:
            print(f"  [Bathymetry] Batch error: {e}")
            return []

    def get_grid(self, center_lat: float, center_lon: float,
                 radius_km: float = 20, resolution_km: float = 2) -> SeabedGrid:
        """
        Get a grid of depth points around a center location.

        Creates a square grid sampling every resolution_km.
        """
        # Calculate grid bounds
        # 1 degree lat ≈ 111 km
        # 1 degree lon ≈ 111 * cos(lat) km
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))

        # Generate grid points
        locations = []
        lat = center_lat - lat_offset
        while lat <= center_lat + lat_offset:
            lon = center_lon - lon_offset
            while lon <= center_lon + lon_offset:
                locations.append((lat, lon))
                lon += resolution_km / (111.0 * math.cos(math.radians(center_lat)))
            lat += resolution_km / 111.0

        # Fetch depths
        points = self.get_depths_batch(locations)

        if not points:
            return SeabedGrid(center_lat, center_lon, resolution_km)

        # Compute statistics
        depths = [p.depth_m for p in points]
        min_depth = min(depths)
        max_depth = max(depths)
        mean_depth = sum(depths) / len(depths)

        # Variance
        variance = sum((d - mean_depth) ** 2 for d in depths) / len(depths)

        # Gradient (depth change per km)
        # Simple approximation: max - min over diameter
        gradient = abs(max_depth - min_depth) / (radius_km * 2)

        # Canyon detection: deep point near shallow water
        canyon_nearby = min_depth < -500 and max_depth > -200

        # Upwelling likely: steep gradient + deep water
        upwelling_likely = gradient > 50 and min_depth < -300

        return SeabedGrid(
            center_lat=center_lat,
            center_lon=center_lon,
            resolution_km=resolution_km,
            points=points,
            min_depth=min_depth,
            max_depth=max_depth,
            mean_depth=mean_depth,
            depth_variance=variance,
            gradient_magnitude=gradient,
            canyon_nearby=canyon_nearby,
            upwelling_likely=upwelling_likely,
        )


# =============================================================================
# SPATIAL BELIEF ENGINE
# =============================================================================

class SpatialBeliefEngine:
    """
    Creates Bayesian beliefs about spatial regions that explain observations.

    Example beliefs:
    - "Monterey region has canyon upwelling" (explains cold temps)
    - "SF region is shallow shelf" (explains warmer temps)
    - "Half Moon Bay has moderate depth gradient" (explains wave behavior)

    These beliefs are used to ADJUST outlier thresholds:
    - If we EXPECT cold water due to canyon, a cold reading isn't an outlier
    - If we see warm water in canyon zone, THAT'S the outlier
    """

    # Known California coast features
    KNOWN_FEATURES = {
        "monterey_canyon": {
            "lat": 36.79,
            "lon": -122.47,
            "type": "submarine_canyon",
            "effects": ["cold_upwelling", "high_nutrients", "deep_fish_species"],
            "affected_metrics": ["water_temp", "wave_height"],
        },
        "farallon_escarpment": {
            "lat": 37.7,
            "lon": -123.0,
            "type": "continental_slope",
            "effects": ["cold_water_intrusion", "whale_feeding"],
            "affected_metrics": ["water_temp"],
        },
        "half_moon_bay_shelf": {
            "lat": 37.356,
            "lon": -122.88,
            "type": "continental_shelf",
            "effects": ["moderate_temps", "kelp_forest"],
            "affected_metrics": ["water_temp", "wave_height"],
        },
        "sf_bay_mouth": {
            "lat": 37.75,
            "lon": -122.84,
            "type": "estuary_influence",
            "effects": ["river_mixing", "variable_salinity"],
            "affected_metrics": ["water_temp", "salinity"],
        },
    }

    def __init__(self, truth_layer: TruthLayer):
        self.truth = truth_layer
        self.client = BathymetryClient()
        self.grids: Dict[str, SeabedGrid] = {}

        # Initialize beliefs from known features
        self._init_known_beliefs()

    def _init_known_beliefs(self):
        """Create initial beliefs from known coastal features."""
        for feature_id, feature in self.KNOWN_FEATURES.items():
            # Create belief about the feature
            claim_id = f"spatial_{feature_id}"
            effects_str = ", ".join(feature["effects"])

            self.truth.add_claim(
                claim_id,
                f"{feature_id.replace('_', ' ').title()} causes {effects_str}",
                category="spatial_feature"
            )

            # If it's a canyon, it probably causes cold water
            if feature["type"] == "submarine_canyon":
                cold_claim = f"spatial_{feature_id}_cold_water"
                self.truth.add_claim(
                    cold_claim,
                    f"Water near {feature_id.replace('_', ' ')} is colder than surroundings",
                    category="spatial_effect"
                )
                # Strong positive relationship
                self.truth.add_relationship(claim_id, cold_claim, weight=8.0)
                self.truth.validate(cold_claim, "confirm")  # Pre-validate known effect

    def analyze_buoy_location(self, buoy_id: str, lat: float, lon: float) -> Dict:
        """
        Analyze the seabed around a buoy location.

        Returns spatial context that can explain observations.
        """
        print(f"  Analyzing seabed around {buoy_id} ({lat:.2f}, {lon:.2f})...")

        # Get grid around buoy
        grid = self.client.get_grid(lat, lon, radius_km=15, resolution_km=3)
        self.grids[buoy_id] = grid

        # Build context
        context = {
            "buoy_id": buoy_id,
            "location": (lat, lon),
            "local_depth_m": grid.mean_depth,
            "depth_category": SeabedPoint(lat, lon, grid.mean_depth).depth_category,
            "min_depth_nearby": grid.min_depth,
            "max_depth_nearby": grid.max_depth,
            "gradient_m_per_km": grid.gradient_magnitude,
            "canyon_nearby": grid.canyon_nearby,
            "upwelling_likely": grid.upwelling_likely,
            "expected_effects": [],
        }

        # Determine expected effects
        if grid.canyon_nearby:
            context["expected_effects"].append("cold_upwelling")
            context["expected_effects"].append("nutrient_rich")

        if grid.upwelling_likely:
            context["expected_effects"].append("temperature_variability")
            context["expected_effects"].append("fish_aggregation")

        if grid.mean_depth > -100:
            context["expected_effects"].append("warmer_surface")
            context["expected_effects"].append("wave_shoaling")

        # Create belief about this buoy's spatial context
        claim_id = f"spatial_context_{buoy_id}"
        self.truth.add_claim(
            claim_id,
            f"Buoy {buoy_id} at {grid.mean_depth:.0f}m depth, gradient {grid.gradient_magnitude:.0f}m/km",
            category="buoy_spatial_context"
        )

        if grid.canyon_nearby:
            canyon_claim = f"spatial_{buoy_id}_near_canyon"
            self.truth.add_claim(
                canyon_claim,
                f"Buoy {buoy_id} is near a submarine canyon",
                category="spatial_feature"
            )
            self.truth.add_relationship(canyon_claim, claim_id, weight=5.0)

        return context

    def get_temperature_adjustment(self, buoy_id: str) -> float:
        """
        Get expected temperature adjustment based on spatial context.

        Returns offset in °C:
        - Negative = expect colder than regional average
        - Positive = expect warmer
        """
        grid = self.grids.get(buoy_id)
        if not grid:
            return 0.0

        adjustment = 0.0

        # Deep water / canyon = colder
        if grid.min_depth < -500:
            adjustment -= 1.5  # Expect 1.5°C colder

        # Upwelling zone = colder
        if grid.upwelling_likely:
            adjustment -= 1.0

        # Shallow shelf = warmer
        if grid.mean_depth > -100:
            adjustment += 0.5

        return adjustment

    def adjust_outlier_threshold(self, buoy_id: str, metric: str,
                                  value: float, baseline: float) -> Tuple[float, str]:
        """
        Adjust outlier detection based on spatial context.

        Returns:
        - adjusted_baseline: What we should EXPECT given location
        - explanation: Why we adjusted
        """
        grid = self.grids.get(buoy_id)
        if not grid:
            return baseline, "no spatial context"

        explanation_parts = []
        adjusted = baseline

        if metric == "water_temp":
            temp_adj = self.get_temperature_adjustment(buoy_id)
            adjusted = baseline + temp_adj

            if abs(temp_adj) > 0.1:
                if temp_adj < 0:
                    explanation_parts.append(f"canyon/upwelling zone (-{abs(temp_adj):.1f}°C)")
                else:
                    explanation_parts.append(f"shallow shelf (+{temp_adj:.1f}°C)")

        elif metric == "wave_height":
            # Deep water = less wave shoaling
            if grid.mean_depth < -200:
                # Deep water waves are more uniform
                adjusted = baseline * 0.9
                explanation_parts.append("deep water (less shoaling)")
            else:
                # Shallow water amplifies waves
                adjusted = baseline * 1.1
                explanation_parts.append("shallow shelf (wave amplification)")

        explanation = ", ".join(explanation_parts) if explanation_parts else "no adjustment needed"

        return adjusted, explanation

    def explain_observation(self, buoy_id: str, metric: str, value: float) -> str:
        """
        Generate human-readable explanation for an observation.

        This is the "model feel" - why did we see this value?
        """
        grid = self.grids.get(buoy_id)
        if not grid:
            return "No spatial context available"

        lines = [f"Observation: {metric} = {value:.2f} at {buoy_id}"]
        lines.append(f"Location context:")
        lines.append(f"  - Depth: {grid.mean_depth:.0f}m ({SeabedPoint(0, 0, grid.mean_depth).depth_category})")
        lines.append(f"  - Gradient: {grid.gradient_magnitude:.0f}m/km")

        if grid.canyon_nearby:
            lines.append(f"  - CANYON NEARBY: min depth {grid.min_depth:.0f}m")
            if metric == "water_temp":
                lines.append(f"  → Cold upwelling expected (canyon effect)")

        if grid.upwelling_likely:
            lines.append(f"  - UPWELLING ZONE: steep gradient + deep water")
            lines.append(f"  → Temperature variability expected")

        return "\n".join(lines)


# =============================================================================
# INTEGRATION WITH OUTLIER DETECTION
# =============================================================================

class SpatiallyAwareOutlierDetector:
    """
    Outlier detection that accounts for seabed topology.

    Key insight: A reading is only an outlier if it's unexpected
    GIVEN THE SPATIAL CONTEXT.

    - 13.5°C at Monterey (near canyon) = NORMAL (expect cold)
    - 13.5°C at SF (shallow) = COLD OUTLIER (expect warmer)
    """

    def __init__(self, truth_layer: TruthLayer):
        self.truth = truth_layer
        self.spatial = SpatialBeliefEngine(truth_layer)
        self.buoy_contexts: Dict[str, Dict] = {}

    def initialize_buoy(self, buoy_id: str, lat: float, lon: float):
        """Load spatial context for a buoy location."""
        context = self.spatial.analyze_buoy_location(buoy_id, lat, lon)
        self.buoy_contexts[buoy_id] = context
        return context

    def check_outlier_spatial(self, buoy_id: str, metric: str, value: float,
                              baseline: float, stdev: float) -> Dict:
        """
        Check if value is an outlier, accounting for spatial context.

        Returns dict with:
        - is_outlier: bool
        - adjusted_baseline: float
        - z_score: float (using adjusted baseline)
        - explanation: str
        """
        # Get spatial adjustment
        adjusted_baseline, spatial_reason = self.spatial.adjust_outlier_threshold(
            buoy_id, metric, value, baseline
        )

        # Calculate z-score against adjusted baseline
        z_score = (value - adjusted_baseline) / max(stdev, 0.001)

        # Determine if outlier
        is_outlier = abs(z_score) > 2.5

        # Generate explanation
        if is_outlier:
            direction = "above" if z_score > 0 else "below"
            explanation = (f"{metric} is {direction} expected "
                          f"(adjusted for {spatial_reason})")
        else:
            if spatial_reason != "no adjustment needed":
                explanation = f"Within normal range after adjusting for {spatial_reason}"
            else:
                explanation = "Within normal range"

        return {
            "is_outlier": is_outlier,
            "original_baseline": baseline,
            "adjusted_baseline": adjusted_baseline,
            "spatial_adjustment": adjusted_baseline - baseline,
            "z_score": z_score,
            "spatial_reason": spatial_reason,
            "explanation": explanation,
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo seabed topology integration."""
    print("=" * 70)
    print("SEABED TOPOLOGY INTEGRATION")
    print("=" * 70)
    print()

    # Initialize
    truth = TruthLayer("spatial_beliefs.json")
    spatial = SpatialBeliefEngine(truth)

    # Buoy locations
    buoys = {
        "46026": {"name": "San Francisco", "lat": 37.750, "lon": -122.838},
        "46042": {"name": "Monterey", "lat": 36.789, "lon": -122.469},
        "46012": {"name": "Half Moon Bay", "lat": 37.356, "lon": -122.881},
    }

    print("Analyzing seabed topology around each buoy...")
    print()

    for buoy_id, info in buoys.items():
        context = spatial.analyze_buoy_location(buoy_id, info["lat"], info["lon"])

        print(f"BUOY {buoy_id} ({info['name']}):")
        print(f"  Local depth: {context['local_depth_m']:.0f}m ({context['depth_category']})")
        print(f"  Depth range nearby: {context['min_depth_nearby']:.0f}m to {context['max_depth_nearby']:.0f}m")
        print(f"  Gradient: {context['gradient_m_per_km']:.0f}m/km")
        print(f"  Canyon nearby: {context['canyon_nearby']}")
        print(f"  Upwelling likely: {context['upwelling_likely']}")
        print(f"  Expected effects: {', '.join(context['expected_effects']) or 'none'}")
        print()

    # Show how this affects outlier detection
    print("-" * 70)
    print("OUTLIER ADJUSTMENT EXAMPLE:")
    print()

    detector = SpatiallyAwareOutlierDetector(truth)

    # Initialize spatial context
    for buoy_id, info in buoys.items():
        detector.initialize_buoy(buoy_id, info["lat"], info["lon"])

    # Test case: Same temperature reading at different locations
    test_temp = 13.0  # °C
    baseline_temp = 13.5  # Regional average
    stdev_temp = 0.3

    print(f"Test: Water temperature = {test_temp}°C")
    print(f"Regional baseline: {baseline_temp}°C (σ={stdev_temp})")
    print()

    for buoy_id, info in buoys.items():
        result = detector.check_outlier_spatial(
            buoy_id, "water_temp", test_temp, baseline_temp, stdev_temp
        )

        status = "🚨 OUTLIER" if result["is_outlier"] else "✓ Normal"
        print(f"  {buoy_id} ({info['name']}): {status}")
        print(f"    Adjusted baseline: {result['adjusted_baseline']:.2f}°C")
        print(f"    Z-score: {result['z_score']:.1f}")
        print(f"    Reason: {result['explanation']}")
        print()

    # Show TruthLayer state
    print("-" * 70)
    print("SPATIAL BELIEFS IN TRUTHLAYER:")
    print(truth.get_truth_context())

    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("  The SAME temperature reading can be:")
    print("  - Normal near a canyon (cold upwelling expected)")
    print("  - An outlier on the shelf (should be warmer)")
    print()
    print("  Seabed topology is the HIDDEN VARIABLE that explains")
    print("  variations humans struggle to grasp intuitively.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
