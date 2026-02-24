"""
Ocean Data Feeds - Real-Time Buoy & Marine Data
================================================

6 FREE API feeds for San Francisco / Half Moon Bay / Monterey:

1. NDBC Buoy 46026 (SF) - Wave height, period, water temp
2. NDBC Buoy 46012 (Half Moon Bay) - Wave height, period, water temp
3. NOAA Tides (SF Station 9414290) - Tide predictions vs actual
4. Open-Meteo Marine - Wave forecasts (predictions)
5. USGS Earthquakes - Seismic activity (correlates with ocean anomalies)
6. NWS Marine Forecast - Weather predictions

THE IDEA:
---------
1. Ingest real-time buoy data
2. Detect outliers (wave height spike, temp anomaly, etc.)
3. Create PREDICTIONS using TruthLayer belief propagation
4. Compare predictions to ACTUAL outcomes
5. Delta = our error rate = what to improve

If we predict better than baseline (simple moving average), we add value.
"""

import json
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BuoyReading:
    """Single reading from a buoy."""
    station_id: str
    timestamp: datetime
    wave_height: Optional[float] = None      # meters
    wave_period: Optional[float] = None      # seconds
    wave_direction: Optional[int] = None     # degrees
    water_temp: Optional[float] = None       # celsius
    wind_speed: Optional[float] = None       # m/s
    wind_gust: Optional[float] = None        # m/s
    pressure: Optional[float] = None         # hPa


@dataclass
class TidePrediction:
    """Tide prediction from NOAA."""
    station_id: str
    timestamp: datetime
    predicted_level: float  # feet
    actual_level: Optional[float] = None


@dataclass
class Outlier:
    """Detected outlier event."""
    source: str              # buoy_46026, tide_sf, etc.
    metric: str              # wave_height, water_temp, etc.
    timestamp: datetime
    value: float
    baseline: float          # what we expected
    z_score: float           # how many std devs from mean
    severity: str            # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class Prediction:
    """Our prediction vs what actually happened."""
    prediction_id: str
    source: str
    metric: str
    predicted_at: datetime
    prediction_time: datetime    # when we predicted FOR
    predicted_value: float
    predicted_direction: str     # UP, DOWN, STABLE
    confidence: float            # 0-1

    # Filled in later when we have actuals
    actual_value: Optional[float] = None
    error: Optional[float] = None
    correct_direction: Optional[bool] = None


# =============================================================================
# API CLIENTS
# =============================================================================

class NDBCClient:
    """
    National Data Buoy Center - Real buoy data.

    Stations:
    - 46026: San Francisco (37.75N, 122.84W)
    - 46012: Half Moon Bay (37.36N, 122.88W)
    - 46042: Monterey (36.79N, 122.47W)
    """

    BASE_URL = "https://www.ndbc.noaa.gov/data/realtime2"

    STATIONS = {
        "46026": {"name": "San Francisco", "lat": 37.750, "lon": -122.838},
        "46012": {"name": "Half Moon Bay", "lat": 37.356, "lon": -122.881},
        "46042": {"name": "Monterey", "lat": 36.789, "lon": -122.469},
    }

    def fetch_latest(self, station_id: str, hours: int = 24) -> List[BuoyReading]:
        """Fetch latest readings from a buoy station."""
        url = f"{self.BASE_URL}/{station_id}.txt"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode('utf-8')
        except urllib.error.URLError as e:
            print(f"  [NDBC] Error fetching {station_id}: {e}")
            return []

        readings = []
        lines = text.strip().split('\n')

        # Skip header lines (start with #)
        data_lines = [l for l in lines if not l.startswith('#')]

        for line in data_lines[:hours]:  # Limit to requested hours
            parts = line.split()
            if len(parts) < 14:
                continue

            try:
                # Parse timestamp
                year, month, day, hour, minute = map(int, parts[0:5])
                ts = datetime(year, month, day, hour, minute)

                # Parse values (MM = missing)
                def parse_val(s: str) -> Optional[float]:
                    if s == 'MM':
                        return None
                    try:
                        return float(s)
                    except:
                        return None

                reading = BuoyReading(
                    station_id=station_id,
                    timestamp=ts,
                    wind_speed=parse_val(parts[6]),
                    wind_gust=parse_val(parts[7]),
                    wave_height=parse_val(parts[8]),
                    wave_period=parse_val(parts[9]),
                    wave_direction=int(parts[11]) if parts[11] != 'MM' else None,
                    pressure=parse_val(parts[12]),
                    water_temp=parse_val(parts[14]),
                )
                readings.append(reading)

            except (ValueError, IndexError) as e:
                continue

        return readings


class NOAATidesClient:
    """
    NOAA CO-OPS Tides and Currents API.

    Stations:
    - 9414290: San Francisco
    - 9414131: Point Reyes
    - 9413450: Monterey
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    STATIONS = {
        "9414290": {"name": "San Francisco", "lat": 37.807, "lon": -122.465},
        "9414131": {"name": "Point Reyes", "lat": 37.996, "lon": -122.976},
        "9413450": {"name": "Monterey", "lat": 36.605, "lon": -121.888},
    }

    def fetch_predictions(self, station_id: str, hours: int = 24) -> List[TidePrediction]:
        """Fetch tide predictions."""
        params = {
            "station": station_id,
            "product": "predictions",
            "datum": "MLLW",
            "time_zone": "gmt",
            "units": "english",
            "format": "json",
            "date": "today",
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  [Tides] Error fetching {station_id}: {e}")
            return []

        predictions = []
        for p in data.get("predictions", [])[:hours * 10]:  # ~6 min intervals
            try:
                ts = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
                predictions.append(TidePrediction(
                    station_id=station_id,
                    timestamp=ts,
                    predicted_level=float(p["v"]),
                ))
            except (ValueError, KeyError):
                continue

        return predictions

    def fetch_actual(self, station_id: str, hours: int = 6) -> List[TidePrediction]:
        """Fetch actual water levels (to compare with predictions)."""
        params = {
            "station": station_id,
            "product": "water_level",
            "datum": "MLLW",
            "time_zone": "gmt",
            "units": "english",
            "format": "json",
            "date": "today",
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  [Tides] Error fetching actuals {station_id}: {e}")
            return []

        readings = []
        for p in data.get("data", []):
            try:
                ts = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
                readings.append(TidePrediction(
                    station_id=station_id,
                    timestamp=ts,
                    predicted_level=0,  # Not a prediction
                    actual_level=float(p["v"]),
                ))
            except (ValueError, KeyError):
                continue

        return readings


class OpenMeteoClient:
    """
    Open-Meteo Marine API - Wave forecasts.
    Free, no API key needed.
    """

    BASE_URL = "https://marine-api.open-meteo.com/v1/marine"

    def fetch_forecast(self, lat: float, lon: float) -> Dict:
        """Fetch 7-day wave forecast."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period",
            "timezone": "America/Los_Angeles",
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  [OpenMeteo] Error: {e}")
            return {}


class USGSEarthquakeClient:
    """
    USGS Earthquake API.
    Seismic activity can correlate with ocean anomalies.
    """

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def fetch_recent(self, lat: float, lon: float, radius_km: int = 200,
                     min_magnitude: float = 1.0) -> List[Dict]:
        """Fetch recent earthquakes near a location."""
        params = {
            "format": "geojson",
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": radius_km,
            "minmagnitude": min_magnitude,
            "orderby": "time",
            "limit": 20,
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get("features", [])
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  [USGS] Error: {e}")
            return []


class NWSMarineClient:
    """
    National Weather Service API - Marine forecasts.
    Free, no API key needed.
    """

    def fetch_forecast(self, office: str, grid_x: int, grid_y: int) -> Dict:
        """Fetch marine forecast for a grid point."""
        url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanBuoyApp/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"  [NWS] Error: {e}")
            return {}


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

class OutlierDetector:
    """
    Detect outliers in ocean data streams.

    Uses rolling statistics to identify anomalies.
    """

    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}

    def _get_key(self, source: str, metric: str) -> str:
        return f"{source}:{metric}"

    def add_value(self, source: str, metric: str, value: float):
        """Add a value to the rolling history."""
        key = self._get_key(source, metric)
        if key not in self.history:
            self.history[key] = []

        self.history[key].append(value)

        # Keep only window_size values
        if len(self.history[key]) > self.window_size * 2:
            self.history[key] = self.history[key][-self.window_size:]

    def check_outlier(self, source: str, metric: str, value: float,
                      timestamp: datetime) -> Optional[Outlier]:
        """Check if a value is an outlier. Returns Outlier if yes, None if no."""
        key = self._get_key(source, metric)
        history = self.history.get(key, [])

        if len(history) < 5:  # Need enough history
            return None

        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0.1

        if stdev < 0.01:  # Avoid division by tiny numbers
            stdev = 0.01

        z_score = (value - mean) / stdev

        # Classify severity
        abs_z = abs(z_score)
        if abs_z < 2.0:
            return None  # Not an outlier
        elif abs_z < 2.5:
            severity = "LOW"
        elif abs_z < 3.0:
            severity = "MEDIUM"
        elif abs_z < 4.0:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        return Outlier(
            source=source,
            metric=metric,
            timestamp=timestamp,
            value=value,
            baseline=mean,
            z_score=z_score,
            severity=severity,
        )


# =============================================================================
# OCEAN DATA AGGREGATOR
# =============================================================================

class OceanDataAggregator:
    """
    Main class that pulls from all feeds and detects outliers.
    """

    def __init__(self):
        self.ndbc = NDBCClient()
        self.tides = NOAATidesClient()
        self.meteo = OpenMeteoClient()
        self.usgs = USGSEarthquakeClient()
        self.nws = NWSMarineClient()
        self.detector = OutlierDetector()

        # Storage
        self.readings: List[BuoyReading] = []
        self.outliers: List[Outlier] = []
        self.predictions: List[Prediction] = []

    def fetch_all(self) -> Dict:
        """Fetch data from all sources."""
        print("Fetching ocean data from 6 API feeds...")

        results = {
            "buoy_readings": [],
            "tide_data": [],
            "wave_forecast": {},
            "earthquakes": [],
            "weather_forecast": {},
            "outliers": [],
        }

        # 1. NDBC Buoys
        for station_id in ["46026", "46012"]:
            print(f"  [1] NDBC Buoy {station_id}...")
            readings = self.ndbc.fetch_latest(station_id, hours=24)
            results["buoy_readings"].extend(readings)

            # Check for outliers in wave height
            for r in readings:
                if r.wave_height is not None:
                    self.detector.add_value(f"buoy_{station_id}", "wave_height", r.wave_height)
                    outlier = self.detector.check_outlier(
                        f"buoy_{station_id}", "wave_height", r.wave_height, r.timestamp
                    )
                    if outlier:
                        results["outliers"].append(outlier)

                if r.water_temp is not None:
                    self.detector.add_value(f"buoy_{station_id}", "water_temp", r.water_temp)
                    outlier = self.detector.check_outlier(
                        f"buoy_{station_id}", "water_temp", r.water_temp, r.timestamp
                    )
                    if outlier:
                        results["outliers"].append(outlier)

        # 2. NOAA Tides
        print("  [2] NOAA Tides SF...")
        predictions = self.tides.fetch_predictions("9414290")
        actuals = self.tides.fetch_actual("9414290")
        results["tide_data"] = {"predictions": predictions, "actuals": actuals}

        # 3. Open-Meteo Marine Forecast
        print("  [3] Open-Meteo Marine Forecast...")
        results["wave_forecast"] = self.meteo.fetch_forecast(37.5, -122.5)

        # 4. USGS Earthquakes
        print("  [4] USGS Earthquakes...")
        results["earthquakes"] = self.usgs.fetch_recent(37.5, -122.5)

        # 5. NWS Marine Forecast
        print("  [5] NWS Marine Forecast...")
        results["weather_forecast"] = self.nws.fetch_forecast("MTR", 84, 105)

        print(f"  [6] Outlier detection...")
        self.outliers.extend(results["outliers"])

        return results

    def summarize(self, results: Dict) -> str:
        """Create human-readable summary."""
        lines = [
            "=" * 60,
            "OCEAN DATA SUMMARY",
            "=" * 60,
            "",
            f"Buoy readings: {len(results['buoy_readings'])}",
        ]

        # Wave stats
        wave_heights = [r.wave_height for r in results['buoy_readings']
                       if r.wave_height is not None]
        if wave_heights:
            lines.append(f"  Wave height: {min(wave_heights):.1f}m - {max(wave_heights):.1f}m (avg: {statistics.mean(wave_heights):.1f}m)")

        water_temps = [r.water_temp for r in results['buoy_readings']
                      if r.water_temp is not None]
        if water_temps:
            lines.append(f"  Water temp: {min(water_temps):.1f}°C - {max(water_temps):.1f}°C")

        # Tides
        tide_preds = results['tide_data'].get('predictions', [])
        if tide_preds:
            levels = [t.predicted_level for t in tide_preds]
            lines.append(f"Tide predictions: {len(tide_preds)} points ({min(levels):.1f}ft - {max(levels):.1f}ft)")

        # Earthquakes
        quakes = results['earthquakes']
        if quakes:
            mags = [q['properties']['mag'] for q in quakes]
            lines.append(f"Recent earthquakes: {len(quakes)} (max mag: {max(mags):.1f})")

        # Outliers
        lines.append("")
        lines.append(f"OUTLIERS DETECTED: {len(results['outliers'])}")
        for o in results['outliers'][:5]:
            lines.append(f"  [{o.severity}] {o.source}/{o.metric}: {o.value:.2f} (expected: {o.baseline:.2f}, z={o.z_score:.1f})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# TRUTHLAYER INTEGRATION
# =============================================================================

def create_ocean_beliefs(truth_layer, outliers: List[Outlier]):
    """
    Convert ocean outliers into TruthLayer beliefs.

    The key insight: When we detect an outlier, we create PREDICTIONS
    about what will happen next. Then we validate against actuals.
    """
    from truth_layer import TruthLayer

    for outlier in outliers:
        # Create belief about the outlier
        claim_id = f"outlier_{outlier.source}_{outlier.metric}_{outlier.timestamp.strftime('%Y%m%d%H%M')}"

        text = f"{outlier.source} {outlier.metric} anomaly: {outlier.value:.2f} (z={outlier.z_score:.1f})"

        truth_layer.add_claim(claim_id, text, category="ocean_outlier")

        # Create prediction beliefs
        if outlier.z_score > 0:
            # Value is HIGH - predict it will come back down
            pred_id = f"pred_decrease_{claim_id}"
            truth_layer.add_claim(
                pred_id,
                f"{outlier.source} {outlier.metric} will decrease in next 6 hours",
                category="ocean_prediction"
            )
            # Link: outlier being true increases confidence in mean reversion
            truth_layer.add_relationship(claim_id, pred_id, weight=5.0)
        else:
            # Value is LOW - predict it will come back up
            pred_id = f"pred_increase_{claim_id}"
            truth_layer.add_claim(
                pred_id,
                f"{outlier.source} {outlier.metric} will increase in next 6 hours",
                category="ocean_prediction"
            )
            truth_layer.add_relationship(claim_id, pred_id, weight=5.0)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the ocean data system."""
    print("=" * 60)
    print("OCEAN DATA FEEDS - Live Demo")
    print("=" * 60)
    print()

    aggregator = OceanDataAggregator()
    results = aggregator.fetch_all()

    print()
    print(aggregator.summarize(results))

    # Show what we'd feed to TruthLayer
    print()
    print("TRUTHLAYER INTEGRATION:")
    print("-" * 40)

    if results['outliers']:
        print(f"Would create {len(results['outliers'])} outlier beliefs")
        print(f"Would create {len(results['outliers'])} prediction beliefs")
        print()
        print("Example belief chain:")
        o = results['outliers'][0]
        print(f"  1. OUTLIER: {o.source}/{o.metric} = {o.value:.2f}")
        print(f"  2. PREDICTION: Will revert toward {o.baseline:.2f}")
        print(f"  3. VALIDATION: Compare to actual value in 6 hours")
        print(f"  4. LEARNING: If wrong, update belief weights")
    else:
        print("No outliers detected (ocean is calm)")
        print("This is actually useful data - confirms baseline is accurate")

    print()
    print("=" * 60)
    print("DATA SOURCES (all free, no API key):")
    print("  1. NDBC Buoy 46026 (SF)")
    print("  2. NDBC Buoy 46012 (Half Moon Bay)")
    print("  3. NOAA Tides (SF)")
    print("  4. Open-Meteo Marine Forecast")
    print("  5. USGS Earthquakes")
    print("  6. NWS Marine Forecast")
    print("=" * 60)


if __name__ == "__main__":
    demo()
