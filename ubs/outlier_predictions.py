"""
Outlier Prediction Engine
=========================

Connects multi-feed outliers to TruthLayer for:
1. Pattern learning (which outliers correlate?)
2. Prediction (what happens after an outlier?)
3. Validation (were we right?)
4. Value measurement (do we beat baseline?)
5. Spatial context (seabed topology explains hidden variables)

KEY INSIGHT:
------------
Outliers from different feeds may correlate:
- Earthquake → wave height spike (tsunami)
- Solar flare → GPS anomaly → aircraft altitude reporting errors
- Low pressure → high waves
- Low stream flow + high temp → fish die-off
- Seabed canyon → cold upwelling → temperature "anomaly" is actually normal

TruthLayer tracks these correlations and learns which matter.
Seabed topology is the HIDDEN VARIABLE that explains variations
humans struggle to grasp intuitively.
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from truth_layer import TruthLayer
from multi_feed_outliers import (
    MultiFeedAggregator, Outlier, UnifiedOutlierDetector,
    UPDATE_INTERVALS, SF_BAY_BOUNDS
)
from rumor_feeds import ExtendedRumorAggregator


@dataclass
class CrossFeedPrediction:
    """Prediction that one outlier will cause another."""
    prediction_id: str
    trigger_feed: str
    trigger_metric: str
    target_feed: str
    target_metric: str
    predicted_direction: str  # UP, DOWN
    confidence: float
    created_at: datetime
    horizon_hours: int

    # Filled after validation
    validated: bool = False
    correct: Optional[bool] = None


class OutlierPredictionEngine:
    """
    Makes and validates predictions about outlier cascades.

    Example predictions:
    - If wave_height spikes, wave_period will also spike (same feed correlation)
    - If earthquake M>3, wave_height may spike within 2 hours (cross-feed)
    - If Kp index > 5, aircraft altitude readings may be noisy (space weather → aviation)
    """

    # Known correlations to bootstrap the system
    KNOWN_CORRELATIONS = [
        # Same feed correlations
        ("ndbc", "wave_height", "ndbc", "wave_period", 0.7, "UP"),
        ("ndbc", "wind_speed", "ndbc", "wave_height", 0.6, "UP"),
        ("ndbc", "pressure", "ndbc", "wave_height", -0.5, "UP"),  # Low pressure → high waves

        # Cross feed correlations
        ("usgs_eq", "magnitude", "ndbc", "wave_height", 0.3, "UP"),  # Earthquake → waves
        ("space", "kp_index", "opensky", "altitude", 0.2, "NOISY"),  # Solar → GPS errors
        ("air", "pm25", "air", "aqi", 0.9, "UP"),  # PM2.5 → AQI

        # Water correlations
        ("ndbc", "wave_height", "usgs_water", "discharge", 0.4, "UP"),  # Coastal storm
    ]

    def __init__(self, truth_layer: TruthLayer):
        self.truth = truth_layer
        self.predictions: List[CrossFeedPrediction] = []
        self.validated: List[CrossFeedPrediction] = []

        # Initialize correlation beliefs
        self._init_correlation_beliefs()

    def _init_correlation_beliefs(self):
        """Create TruthLayer beliefs for known correlations."""
        for src_feed, src_metric, tgt_feed, tgt_metric, strength, direction in self.KNOWN_CORRELATIONS:
            claim_id = f"corr_{src_feed}_{src_metric}_to_{tgt_feed}_{tgt_metric}"
            text = f"{src_feed}/{src_metric} outlier correlates with {tgt_feed}/{tgt_metric}"

            self.truth.add_claim(claim_id, text, category="correlation")

            # Pre-weight based on known strength
            if strength > 0.5:
                self.truth.validate(claim_id, "confirm")

    def process_outliers(self, outliers: List[Outlier]) -> List[CrossFeedPrediction]:
        """
        Process a batch of outliers and generate predictions.

        For each outlier, check if it should trigger predictions
        about other feeds based on learned correlations.
        """
        new_predictions = []

        for outlier in outliers:
            # Record the outlier as a belief
            outlier_claim = f"outlier_{outlier.feed}_{outlier.metric}_{datetime.now().strftime('%Y%m%d%H%M')}"
            self.truth.add_claim(
                outlier_claim,
                outlier.to_belief_text(),
                category="outlier_event"
            )

            # Check for correlations
            for src_feed, src_metric, tgt_feed, tgt_metric, strength, direction in self.KNOWN_CORRELATIONS:
                # Match if this outlier could trigger a prediction
                if src_feed in outlier.feed and src_metric == outlier.metric:
                    # Check belief strength for this correlation
                    corr_claim = f"corr_{src_feed}_{src_metric}_to_{tgt_feed}_{tgt_metric}"
                    belief = self.truth.get_belief(corr_claim)

                    if belief and belief.probability > 0.4:
                        # Make prediction
                        pred = CrossFeedPrediction(
                            prediction_id=f"pred_{datetime.now().strftime('%H%M%S')}_{tgt_feed}_{tgt_metric}",
                            trigger_feed=outlier.feed,
                            trigger_metric=outlier.metric,
                            target_feed=tgt_feed,
                            target_metric=tgt_metric,
                            predicted_direction=direction,
                            confidence=belief.probability * strength,
                            created_at=datetime.now(),
                            horizon_hours=2,
                        )

                        new_predictions.append(pred)
                        self.predictions.append(pred)

                        # Create prediction belief
                        pred_claim = f"pred_{tgt_feed}_{tgt_metric}_{direction.lower()}"
                        self.truth.add_claim(
                            pred_claim,
                            f"Predicting {tgt_feed}/{tgt_metric} will go {direction} within 2 hours",
                            category="prediction"
                        )

                        # Link to trigger
                        self.truth.add_relationship(outlier_claim, pred_claim, weight=5.0)

        return new_predictions

    def validate_predictions(self, outliers: List[Outlier]) -> List[CrossFeedPrediction]:
        """
        Check if any pending predictions came true.

        An outlier in the target feed/metric validates the prediction.
        """
        validated = []
        now = datetime.now()

        for pred in self.predictions:
            if pred.validated:
                continue

            # Check if prediction window has passed
            window_end = pred.created_at + timedelta(hours=pred.horizon_hours)
            if now < window_end:
                continue  # Still in prediction window

            # Look for matching outliers
            matching = [
                o for o in outliers
                if pred.target_feed in o.feed and pred.target_metric == o.metric
            ]

            pred.validated = True

            if matching:
                # Prediction came true!
                pred.correct = True
                validated.append(pred)

                # Update correlation belief positively
                corr_claim = f"corr_{pred.trigger_feed}_{pred.trigger_metric}_to_{pred.target_feed}_{pred.target_metric}"
                if corr_claim in self.truth.net.beliefs:
                    self.truth.validate(corr_claim, "confirm")

            else:
                # Prediction was wrong
                pred.correct = False
                validated.append(pred)

                # Update correlation belief negatively
                corr_claim = f"corr_{pred.trigger_feed}_{pred.trigger_metric}_to_{pred.target_feed}_{pred.target_metric}"
                if corr_claim in self.truth.net.beliefs:
                    self.truth.validate(corr_claim, "reject")

            self.validated.append(pred)

        return validated

    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        correct = sum(1 for p in self.validated if p.correct) if self.validated else 0
        accuracy = correct / len(self.validated) if self.validated else 0
        pending = len([p for p in self.predictions if not p.validated])

        return {
            "predictions": len(self.predictions),
            "validated": len(self.validated),
            "correct": correct,
            "accuracy": accuracy,
            "pending": pending,
        }


# =============================================================================
# INTEGRATED RUNNER
# =============================================================================

class OutlierMonitor:
    """
    Complete system: Fetch → Detect → Predict → Validate → Learn

    Now with spatial awareness AND rumor/crowdsourced feeds:
    - Loads seabed topology for each buoy location
    - Adjusts outlier baselines based on depth/canyon/upwelling
    - Same reading can be normal at one location, outlier at another
    - Integrates citizen science data (iNaturalist, OBIS, GDELT)
    - Combines official sensor feeds with "human sensor network"
    """

    def __init__(self, truth_path: str = "outlier_beliefs.json",
                 use_spatial: bool = True, use_rumor_feeds: bool = True):
        self.truth = TruthLayer(truth_path)
        self.aggregator = MultiFeedAggregator()
        self.predictor = OutlierPredictionEngine(self.truth)

        # Spatial awareness
        self.use_spatial = use_spatial
        self.spatial_engine = None
        if use_spatial:
            try:
                from seabed_topology import SpatiallyAwareOutlierDetector
                self.spatial_engine = SpatiallyAwareOutlierDetector(self.truth)
                self._init_spatial_contexts()
            except ImportError:
                print("  [Spatial] seabed_topology.py not found, running without spatial context")
                self.use_spatial = False

        # Rumor/crowdsourced feeds (human sensor network)
        self.use_rumor_feeds = use_rumor_feeds
        self.rumor_aggregator = None
        if use_rumor_feeds:
            try:
                self.rumor_aggregator = ExtendedRumorAggregator()
                print("  [Rumor] Crowdsourced feeds enabled (iNaturalist, GDELT, OBIS)")
            except ImportError:
                print("  [Rumor] rumor_feeds.py not found, running without crowdsourced data")
                self.use_rumor_feeds = False

        self.all_outliers: List[Outlier] = []
        self.run_count = 0

    def _init_spatial_contexts(self):
        """Initialize seabed context for known buoy locations."""
        if not self.spatial_engine:
            return

        buoy_locations = {
            "46026": (37.750, -122.838),
            "46042": (36.789, -122.469),
            "46012": (37.356, -122.881),
            "46013": (38.253, -123.303),
            "46014": (39.196, -123.969),
        }

        print("  Loading seabed topology for buoy locations...")
        for buoy_id, (lat, lon) in buoy_locations.items():
            self.spatial_engine.initialize_buoy(buoy_id, lat, lon)

    def run_once(self) -> Dict:
        """Run one fetch-detect-predict cycle."""
        self.run_count += 1

        # Fetch and detect outliers from official sensors
        results = self.aggregator.fetch_all()
        outliers = results["outliers"]
        sensor_data_points = results["stats"]["total_data_points"]

        # Fetch crowdsourced/rumor feeds (human sensor network)
        rumor_data_points = 0
        rumor_alerts = []
        if self.use_rumor_feeds and self.rumor_aggregator:
            print()
            print("Fetching crowdsourced feeds...")
            rumor_results = self.rumor_aggregator.fetch_all_extended()
            rumor_data_points = rumor_results["stats"]["total_observations"]
            rumor_alerts = rumor_results.get("alerts", [])

            # Convert rumor outliers to standard outliers
            for outlier in rumor_results.get("outliers", []):
                outliers.append(outlier)

            # Create beliefs from rumor alerts
            for alert in rumor_alerts:
                if alert["type"] == "news_surge":
                    claim_id = f"news_{alert['theme']}_{datetime.now().strftime('%Y%m%d')}"
                    self.truth.add_claim(
                        claim_id,
                        f"News surge: {alert['theme']} ({alert['count']} mentions)",
                        category="rumor_alert"
                    )
                elif alert["type"] == "cetacean_concentration":
                    claim_id = f"cetacean_{alert['species'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                    self.truth.add_claim(
                        claim_id,
                        f"Cetacean concentration: {alert['species']} ({alert['count']} sightings)",
                        category="wildlife_alert"
                    )

        # Apply spatial filtering to buoy outliers
        spatial_filtered_count = 0
        if self.use_spatial and self.spatial_engine:
            filtered_outliers = []
            for outlier in outliers:
                # Only filter NDBC buoy outliers (they have spatial context)
                if outlier.feed.startswith("ndbc_"):
                    buoy_id = outlier.feed.replace("ndbc_", "")

                    # Recover stdev from z_score: stdev = (value - baseline) / z_score
                    if abs(outlier.z_score) > 0.001:
                        stdev = abs(outlier.value - outlier.baseline) / abs(outlier.z_score)
                    else:
                        stdev = 0.1  # fallback

                    # Check if still outlier after spatial adjustment
                    result = self.spatial_engine.check_outlier_spatial(
                        buoy_id, outlier.metric, outlier.value,
                        outlier.baseline, stdev
                    )

                    if result["is_outlier"]:
                        # Still an outlier after spatial adjustment - keep it
                        outlier.metadata["spatial_adjusted"] = True
                        outlier.metadata["spatial_reason"] = result["spatial_reason"]
                        outlier.metadata["adjusted_baseline"] = result["adjusted_baseline"]
                        filtered_outliers.append(outlier)
                    else:
                        # Spatial context explains this "outlier" - it's actually normal
                        spatial_filtered_count += 1
                        print(f"  [Spatial] Filtered: {outlier.feed}/{outlier.metric} = {outlier.value:.2f}")
                        print(f"            {result['explanation']}")
                else:
                    # Non-buoy outlier - keep as-is
                    filtered_outliers.append(outlier)

            outliers = filtered_outliers

        # Store outliers
        self.all_outliers.extend(outliers)

        # Generate predictions from new outliers
        new_predictions = self.predictor.process_outliers(outliers)

        # Validate any pending predictions
        validated = self.predictor.validate_predictions(self.all_outliers)

        return {
            "run": self.run_count,
            "timestamp": datetime.now().isoformat(),
            "sensor_data_points": sensor_data_points,
            "rumor_data_points": rumor_data_points,
            "total_data_points": sensor_data_points + rumor_data_points,
            "outliers": len(outliers),
            "spatial_filtered": spatial_filtered_count,
            "rumor_alerts": len(rumor_alerts),
            "new_predictions": len(new_predictions),
            "validated_predictions": len(validated),
            "prediction_stats": self.predictor.get_stats(),
        }

    def get_truth_context(self) -> str:
        """Get current TruthLayer state."""
        return self.truth.get_truth_context()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the complete outlier prediction system."""
    print("=" * 70)
    print("UNIFIED OUTLIER PREDICTION ENGINE")
    print("Official Sensors + Crowdsourced Human Network")
    print("=" * 70)
    print()
    print("DATA SOURCES:")
    print("  Official Sensors:")
    print("    - NDBC Buoys (wave, temp, wind, pressure)")
    print("    - USGS Earthquakes & Stream Flow")
    print("    - NOAA Space Weather")
    print("    - OpenSky Aircraft")
    print("    - Air Quality (PM2.5, AQI)")
    print()
    print("  Crowdsourced (Human Sensor Network):")
    print("    - iNaturalist (wildlife sightings)")
    print("    - GDELT (news event monitoring)")
    print("    - OBIS (whale/dolphin via Happywhale)")
    print()

    monitor = OutlierMonitor("demo_outlier_beliefs.json")

    print()
    print("Running fetch cycle...")
    print()

    result = monitor.run_once()

    print()
    print("-" * 70)
    print(f"Run #{result['run']}")
    print(f"  Sensor data points: {result['sensor_data_points']}")
    print(f"  Crowdsourced data points: {result['rumor_data_points']}")
    print(f"  TOTAL data points: {result['total_data_points']}")
    print(f"  Outliers detected: {result['outliers']}")
    if result.get('spatial_filtered', 0) > 0:
        print(f"  Spatial filtered (false positives): {result['spatial_filtered']}")
    print(f"  Rumor alerts: {result['rumor_alerts']}")
    print(f"  Predictions generated: {result['new_predictions']}")
    print()

    if result['outliers'] > 0:
        print("OUTLIERS DETECTED:")
        for o in monitor.all_outliers[:5]:
            print(f"  [{o.severity}] {o.feed}/{o.metric}: {o.value:.2f} (z={o.z_score:.1f})")
        print()

    if result['new_predictions'] > 0:
        print("PREDICTIONS MADE:")
        for p in monitor.predictor.predictions[:5]:
            print(f"  If {p.trigger_feed}/{p.trigger_metric} -> then {p.target_feed}/{p.target_metric} {p.predicted_direction}")
            print(f"     Confidence: {p.confidence:.1%}, Horizon: {p.horizon_hours}h")
        print()

    stats = monitor.predictor.get_stats()
    print("PREDICTION STATISTICS:")
    print(f"  Total predictions: {stats['predictions']}")
    print(f"  Validated: {stats['validated']}")
    print(f"  Pending: {stats['pending']}")
    if stats['validated'] > 0:
        print(f"  Accuracy: {stats['accuracy']:.1%}")

    print()
    print("=" * 70)
    print("UNIFIED VALUE PROPOSITION:")
    print("  1. Official sensors: High accuracy, slow updates")
    print("  2. Crowdsourced data: Fast detection, lower precision")
    print("  3. Combined: Early warning + confirmation")
    print("  4. TruthLayer learns which cross-feed patterns matter")
    print("  5. Predictions improve over time via feedback loop")
    print()
    print("AUDIENCE-SPECIFIC VALUE:")
    print("  Surfers: Wave predictions from buoys + weather news")
    print("  Bird watchers: Migration patterns + ecosystem changes")
    print("  Whale spotters: Cetacean concentrations + ocean conditions")
    print("  Fishermen: Species sightings + water temperature")
    print("  Commercial marine: Vessel activity + weather events")
    print("=" * 70)


if __name__ == "__main__":
    demo()
