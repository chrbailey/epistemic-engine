"""
Prediction Validator
====================

The feedback loop that proves (or disproves) our value.

PROCESS:
1. Make predictions based on outlier detection + TruthLayer beliefs
2. Wait for actual data to arrive
3. Compare predictions to actuals
4. Calculate error metrics
5. Update TruthLayer beliefs based on results
6. Repeat - system gets better over time

VALUE CALCULATION:
- Baseline: Simple moving average prediction
- Our prediction: TruthLayer-informed prediction
- Delta = our_error - baseline_error
- If delta < 0, we add value
"""

import json
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from truth_layer import TruthLayer
from ocean_feeds import (
    OceanDataAggregator, Outlier, BuoyReading,
    NDBCClient, OpenMeteoClient
)


@dataclass
class PredictionResult:
    """Result of comparing prediction to actual."""
    prediction_id: str
    source: str
    metric: str
    predicted_at: datetime
    target_time: datetime

    predicted_value: float
    baseline_value: float    # Simple moving average
    actual_value: float

    our_error: float         # abs(predicted - actual)
    baseline_error: float    # abs(baseline - actual)
    improvement: float       # baseline_error - our_error (positive = we did better)

    correct_direction: bool  # Did we get the trend right?


@dataclass
class ValidationSession:
    """A session of predictions and validations."""
    session_id: str
    started: datetime
    predictions: List[Dict] = field(default_factory=list)
    results: List[PredictionResult] = field(default_factory=list)

    # Aggregate metrics
    total_predictions: int = 0
    validated: int = 0
    our_mae: float = 0.0     # Mean absolute error
    baseline_mae: float = 0.0
    improvement_pct: float = 0.0
    direction_accuracy: float = 0.0


class PredictionValidator:
    """
    Make predictions and validate against actuals.

    This is where we prove our value.
    """

    def __init__(self, truth_layer: TruthLayer):
        self.truth = truth_layer
        self.ndbc = NDBCClient()
        self.meteo = OpenMeteoClient()

        self.sessions: List[ValidationSession] = []
        self.current_session: Optional[ValidationSession] = None

        # Historical data for baseline
        self.history: Dict[str, List[Tuple[datetime, float]]] = {}

    def start_session(self) -> ValidationSession:
        """Start a new validation session."""
        session = ValidationSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started=datetime.now(),
        )
        self.sessions.append(session)
        self.current_session = session
        return session

    def _get_baseline(self, source: str, metric: str, window: int = 6) -> float:
        """Get baseline prediction (simple moving average)."""
        key = f"{source}:{metric}"
        history = self.history.get(key, [])

        if len(history) < 2:
            return 0.0

        # Use last N values
        recent = [v for _, v in history[-window:]]
        return statistics.mean(recent)

    def _get_trend(self, source: str, metric: str) -> str:
        """Get recent trend direction."""
        key = f"{source}:{metric}"
        history = self.history.get(key, [])

        if len(history) < 3:
            return "STABLE"

        recent = [v for _, v in history[-3:]]
        if recent[-1] > recent[0] * 1.05:
            return "UP"
        elif recent[-1] < recent[0] * 0.95:
            return "DOWN"
        return "STABLE"

    def add_observation(self, source: str, metric: str, value: float,
                       timestamp: datetime):
        """Add an observation to history."""
        key = f"{source}:{metric}"
        if key not in self.history:
            self.history[key] = []

        self.history[key].append((timestamp, value))

        # Keep last 48 hours
        cutoff = datetime.now() - timedelta(hours=48)
        self.history[key] = [(t, v) for t, v in self.history[key] if t > cutoff]

    def make_prediction(self, outlier: Outlier, horizon_hours: int = 6) -> Dict:
        """
        Make a prediction based on an outlier.

        This is where TruthLayer comes in:
        - We look at beliefs about this metric
        - We factor in related beliefs (earthquakes, weather, etc.)
        - We make an informed prediction
        """
        source = outlier.source
        metric = outlier.metric

        # Baseline prediction (what a dumb system would predict)
        baseline = self._get_baseline(source, metric)

        # Our prediction factors:
        # 1. Mean reversion (outliers tend to revert)
        mean_reversion_strength = min(abs(outlier.z_score) / 4.0, 1.0)
        reversion_target = outlier.baseline

        # 2. Trend continuation (if there's momentum)
        trend = self._get_trend(source, metric)
        trend_factor = 0.0
        if trend == "UP":
            trend_factor = 0.1  # Slight upward bias
        elif trend == "DOWN":
            trend_factor = -0.1

        # 3. TruthLayer beliefs
        # Check if we have beliefs about this source/metric
        belief_factor = 0.0
        belief_key = f"{source}_{metric}_volatile"
        belief = self.truth.get_belief(belief_key)
        if belief and belief.probability > 0.7:
            # High volatility belief - less mean reversion
            mean_reversion_strength *= 0.5

        # Combine factors
        our_prediction = (
            outlier.value * (1 - mean_reversion_strength) +
            reversion_target * mean_reversion_strength +
            trend_factor * outlier.baseline
        )

        # Direction prediction
        if outlier.z_score > 1.5:
            predicted_direction = "DOWN"  # Expect mean reversion
        elif outlier.z_score < -1.5:
            predicted_direction = "UP"
        else:
            predicted_direction = trend

        prediction = {
            "id": f"pred_{source}_{metric}_{datetime.now().strftime('%H%M%S')}",
            "source": source,
            "metric": metric,
            "predicted_at": datetime.now().isoformat(),
            "target_time": (datetime.now() + timedelta(hours=horizon_hours)).isoformat(),
            "horizon_hours": horizon_hours,
            "current_value": outlier.value,
            "baseline_prediction": baseline,
            "our_prediction": our_prediction,
            "predicted_direction": predicted_direction,
            "confidence": 0.5 + mean_reversion_strength * 0.3,
            "factors": {
                "mean_reversion": mean_reversion_strength,
                "trend": trend,
                "belief_adjustment": belief_factor,
            }
        }

        if self.current_session:
            self.current_session.predictions.append(prediction)
            self.current_session.total_predictions += 1

        # Create TruthLayer belief about this prediction
        pred_claim = f"pred_{source}_{metric}_{predicted_direction.lower()}"
        self.truth.add_claim(
            pred_claim,
            f"{source} {metric} will go {predicted_direction} in next {horizon_hours} hours",
            category="ocean_prediction"
        )

        return prediction

    def validate_prediction(self, prediction: Dict, actual_value: float) -> PredictionResult:
        """
        Validate a prediction against actual data.

        This is where we measure our value.
        """
        our_error = abs(prediction["our_prediction"] - actual_value)
        baseline_error = abs(prediction["baseline_prediction"] - actual_value)
        improvement = baseline_error - our_error

        # Check direction
        current = prediction["current_value"]
        predicted_dir = prediction["predicted_direction"]
        actual_dir = "UP" if actual_value > current else ("DOWN" if actual_value < current else "STABLE")
        correct_direction = (predicted_dir == actual_dir) or (predicted_dir == "STABLE" and abs(actual_value - current) < current * 0.02)

        result = PredictionResult(
            prediction_id=prediction["id"],
            source=prediction["source"],
            metric=prediction["metric"],
            predicted_at=datetime.fromisoformat(prediction["predicted_at"]),
            target_time=datetime.fromisoformat(prediction["target_time"]),
            predicted_value=prediction["our_prediction"],
            baseline_value=prediction["baseline_prediction"],
            actual_value=actual_value,
            our_error=our_error,
            baseline_error=baseline_error,
            improvement=improvement,
            correct_direction=correct_direction,
        )

        if self.current_session:
            self.current_session.results.append(result)
            self.current_session.validated += 1

        # Update TruthLayer based on result
        self._update_beliefs(prediction, result)

        return result

    def _update_beliefs(self, prediction: Dict, result: PredictionResult):
        """Update TruthLayer beliefs based on prediction outcome."""

        # Find the prediction claim
        pred_claim = f"pred_{result.source}_{result.metric}_{prediction['predicted_direction'].lower()}"

        if result.correct_direction:
            # We got it right - increase confidence
            self.truth.validate(pred_claim, "confirm")
        else:
            # We got it wrong - decrease confidence
            self.truth.validate(pred_claim, "reject")

        # Update volatility belief
        vol_claim = f"{result.source}_{result.metric}_volatile"
        if vol_claim not in self.truth.net.beliefs:
            self.truth.add_claim(
                vol_claim,
                f"{result.source} {result.metric} is highly volatile",
                category="ocean_pattern"
            )

        # If our mean reversion prediction was wrong, increase volatility belief
        if result.improvement < 0:  # We did worse than baseline
            self.truth.validate(vol_claim, "confirm")

    def get_session_stats(self) -> Dict:
        """Calculate aggregate statistics for current session."""
        if not self.current_session or not self.current_session.results:
            return {"error": "No results yet"}

        results = self.current_session.results

        our_errors = [r.our_error for r in results]
        baseline_errors = [r.baseline_error for r in results]
        improvements = [r.improvement for r in results]
        directions = [r.correct_direction for r in results]

        our_mae = statistics.mean(our_errors)
        baseline_mae = statistics.mean(baseline_errors)

        improvement_pct = ((baseline_mae - our_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
        direction_accuracy = sum(directions) / len(directions) * 100

        self.current_session.our_mae = our_mae
        self.current_session.baseline_mae = baseline_mae
        self.current_session.improvement_pct = improvement_pct
        self.current_session.direction_accuracy = direction_accuracy

        return {
            "session_id": self.current_session.session_id,
            "total_predictions": self.current_session.total_predictions,
            "validated": self.current_session.validated,
            "our_mae": our_mae,
            "baseline_mae": baseline_mae,
            "improvement_pct": improvement_pct,
            "direction_accuracy": direction_accuracy,
            "value_added": improvement_pct > 0,
        }


# =============================================================================
# SIMULATION (since we can't wait 6 hours)
# =============================================================================

def simulate_validation():
    """
    Simulate the prediction/validation loop.

    Since we can't wait 6 hours for real validation, we:
    1. Fetch historical data
    2. Pretend we made predictions at T-6h
    3. Validate against data at T
    4. Show what the feedback loop looks like
    """
    print("=" * 60)
    print("PREDICTION VALIDATION SIMULATION")
    print("=" * 60)
    print()

    # Initialize
    truth = TruthLayer("ocean_predictions.json")
    validator = PredictionValidator(truth)
    ndbc = NDBCClient()

    # Fetch real data
    print("Fetching real buoy data...")
    readings = ndbc.fetch_latest("46026", hours=24)

    if len(readings) < 12:
        print("Not enough data for simulation")
        return

    print(f"Got {len(readings)} readings")
    print()

    # Start session
    session = validator.start_session()

    # Filter to only readings with wave data
    readings_with_waves = [r for r in readings if r.wave_height is not None]
    print(f"Readings with wave data: {len(readings_with_waves)}")

    if len(readings_with_waves) < 8:
        print("Not enough wave data for simulation")
        return

    # Use first 6 readings as "history", rest as older history + future
    history = readings_with_waves[3:]   # Older data
    future = readings_with_waves[:3]    # Recent data

    # Build history
    for r in reversed(history):
        if r.wave_height is not None:
            validator.add_observation("buoy_46026", "wave_height", r.wave_height, r.timestamp)
        if r.water_temp is not None:
            validator.add_observation("buoy_46026", "water_temp", r.water_temp, r.timestamp)

    # Create simulated outliers from the data at T-6h
    t_minus_6 = history[0]  # Most recent "historical" reading (with wave data)

    print("SIMULATION: Pretending it's 6 hours ago...")
    print(f"  Time: {t_minus_6.timestamp}")
    print(f"  Wave height: {t_minus_6.wave_height}m")
    print(f"  Water temp: {t_minus_6.water_temp}°C")
    print()

    # Calculate what baseline would be at T-6h
    wave_history = [r.wave_height for r in history if r.wave_height is not None]
    baseline_wave = statistics.mean(wave_history) if wave_history else 1.0
    stdev_wave = statistics.stdev(wave_history) if len(wave_history) > 1 else 0.1

    if t_minus_6.wave_height:
        z_score = (t_minus_6.wave_height - baseline_wave) / max(stdev_wave, 0.01)

        # Create outlier (even if mild, for demo)
        outlier = Outlier(
            source="buoy_46026",
            metric="wave_height",
            timestamp=t_minus_6.timestamp,
            value=t_minus_6.wave_height,
            baseline=baseline_wave,
            z_score=z_score,
            severity="LOW" if abs(z_score) < 2 else "MEDIUM",
        )

        print("Creating prediction based on data at T-6h...")
        prediction = validator.make_prediction(outlier, horizon_hours=6)

        print(f"  Current value: {prediction['current_value']:.2f}m")
        print(f"  Baseline prediction: {prediction['baseline_prediction']:.2f}m")
        print(f"  Our prediction: {prediction['our_prediction']:.2f}m")
        print(f"  Direction: {prediction['predicted_direction']}")
        print(f"  Confidence: {prediction['confidence']:.1%}")
        print()

        # Now "validate" against actual data at T
        actual = future[-1]  # Most recent reading
        print(f"ACTUAL (at T): {actual.wave_height}m @ {actual.timestamp}")
        print()

        if actual.wave_height:
            result = validator.validate_prediction(prediction, actual.wave_height)

            print("VALIDATION RESULT:")
            print(f"  Our error: {result.our_error:.3f}m")
            print(f"  Baseline error: {result.baseline_error:.3f}m")
            print(f"  Improvement: {result.improvement:.3f}m")
            if result.improvement > 0:
                print(f"  ✅ WE DID BETTER than baseline")
            else:
                print(f"  ❌ Baseline was better")
            print(f"  Direction correct: {'✅' if result.correct_direction else '❌'}")

    # Show session stats
    print()
    print("-" * 40)
    stats = validator.get_session_stats()
    print("SESSION STATISTICS:")
    print(f"  Predictions made: {stats.get('total_predictions', 0)}")
    print(f"  Validated: {stats.get('validated', 0)}")
    print(f"  Our MAE: {stats.get('our_mae', 0):.4f}")
    print(f"  Baseline MAE: {stats.get('baseline_mae', 0):.4f}")
    print(f"  Improvement: {stats.get('improvement_pct', 0):.1f}%")
    print(f"  Direction accuracy: {stats.get('direction_accuracy', 0):.0f}%")
    print()

    if stats.get('improvement_pct', 0) > 0:
        print("🎯 VALUE DEMONSTRATED: We beat naive baseline")
    else:
        print("📊 More data needed to prove value")

    print()
    print("=" * 60)
    print("HOW THIS IMPROVES:")
    print("  1. Each validation updates TruthLayer beliefs")
    print("  2. Wrong predictions → adjust mean reversion strength")
    print("  3. Pattern beliefs emerge (e.g., 'morning readings volatile')")
    print("  4. Over time, predictions get better")
    print("=" * 60)


if __name__ == "__main__":
    simulate_validation()
