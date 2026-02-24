"""
Multi-Feed Outlier Detection System
====================================

Scale outlier detection across diverse data feeds.
Only events that shouldn't be there. At scale.

VERIFIED WORKING APIS (December 2025):
--------------------------------------
1. NDBC Buoys (wave, temp, wind) - 10 min updates
2. NOAA Tides (predictions vs actual) - 6 min updates
3. Open-Meteo Marine (wave forecasts) - hourly
4. USGS Earthquakes - real-time
5. NWS Weather Forecasts - hourly
6. OpenSky Aircraft (171 planes in SF area!) - 10 sec updates
7. NOAA Space Weather (solar flares, geomag) - varies
8. USGS Water (stream flow) - 15 min updates
9. Open-Meteo Air Quality (PM2.5, AQI) - hourly

PLANNED (need auth or WebSocket):
---------------------------------
- AIS Ship Tracking (AISStream.io) - WebSocket
- Whale Sightings (API appears down)
- Sentinel Satellite (needs registration)

ARCHITECTURE:
-------------
Each feed has:
1. Client - fetches raw data
2. Parser - extracts metrics
3. Detector - identifies outliers
4. Scheduler - runs at appropriate interval

TruthLayer receives ONLY outliers, not all data.
"""

import json
import statistics
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import urllib.request
import urllib.error
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

# Update frequencies (seconds)
UPDATE_INTERVALS = {
    "ndbc_buoy": 600,       # 10 minutes
    "noaa_tides": 360,      # 6 minutes
    "open_meteo_marine": 3600,  # 1 hour
    "usgs_earthquake": 60,  # 1 minute (they update frequently)
    "nws_forecast": 3600,   # 1 hour
    "opensky_aircraft": 10, # 10 seconds (but we'll sample less)
    "space_weather": 300,   # 5 minutes
    "usgs_water": 900,      # 15 minutes
    "air_quality": 3600,    # 1 hour
}

# SF Bay Area bounding box
SF_BAY_BOUNDS = {
    "lat_min": 36.5,
    "lat_max": 38.5,
    "lon_min": -123.5,
    "lon_max": -121.5,
}


# =============================================================================
# UNIFIED OUTLIER STRUCTURE
# =============================================================================

@dataclass
class Outlier:
    """Universal outlier event from any feed."""
    feed: str              # opensky, ndbc, usgs_eq, etc.
    metric: str            # altitude, wave_height, magnitude, etc.
    timestamp: datetime
    value: float
    baseline: float        # Expected value
    z_score: float         # Standard deviations from mean
    severity: str          # LOW, MEDIUM, HIGH, CRITICAL

    # Context
    location: Optional[Tuple[float, float]] = None  # lat, lon
    metadata: Dict = field(default_factory=dict)

    def to_belief_text(self) -> str:
        """Convert to TruthLayer belief text."""
        loc = f" at {self.location[0]:.2f}, {self.location[1]:.2f}" if self.location else ""
        return f"{self.feed} {self.metric} outlier: {self.value:.2f} (z={self.z_score:.1f}){loc}"


# =============================================================================
# OUTLIER DETECTOR (shared across all feeds)
# =============================================================================

class UnifiedOutlierDetector:
    """
    Detects outliers across all feeds using rolling statistics.

    Key insight: Different feeds have different baseline behaviors.
    Aircraft altitude varies by flight phase.
    Wave height varies by season.
    Stream flow varies by rainfall.

    We maintain separate histories per feed+metric.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}
        self.outlier_count: Dict[str, int] = {}

    def _key(self, feed: str, metric: str) -> str:
        return f"{feed}:{metric}"

    def add(self, feed: str, metric: str, value: float):
        """Add observation to history."""
        key = self._key(feed, metric)
        if key not in self.history:
            self.history[key] = []

        self.history[key].append(value)

        # Trim to window
        if len(self.history[key]) > self.window_size * 2:
            self.history[key] = self.history[key][-self.window_size:]

    def check(self, feed: str, metric: str, value: float,
              timestamp: datetime, location: Tuple[float, float] = None,
              metadata: Dict = None, z_threshold: float = 2.5) -> Optional[Outlier]:
        """
        Check if value is an outlier.
        Returns Outlier if yes, None if within normal range.
        """
        key = self._key(feed, metric)
        history = self.history.get(key, [])

        if len(history) < 10:
            return None  # Not enough history

        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0.1
        stdev = max(stdev, 0.001)  # Avoid division by zero

        z_score = (value - mean) / stdev

        if abs(z_score) < z_threshold:
            return None  # Not an outlier

        # Classify severity
        abs_z = abs(z_score)
        if abs_z < 3.0:
            severity = "LOW"
        elif abs_z < 4.0:
            severity = "MEDIUM"
        elif abs_z < 5.0:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        # Track outlier count
        self.outlier_count[key] = self.outlier_count.get(key, 0) + 1

        return Outlier(
            feed=feed,
            metric=metric,
            timestamp=timestamp,
            value=value,
            baseline=mean,
            z_score=z_score,
            severity=severity,
            location=location,
            metadata=metadata or {},
        )


# =============================================================================
# FEED CLIENTS
# =============================================================================

class OpenSkyClient:
    """
    OpenSky Network - Real-time aircraft positions.

    Free tier: 10 second resolution, 4000 credits/day
    Updates: Every 10 seconds

    Outliers to detect:
    - Unusual altitude (too low, too high)
    - Unusual speed
    - Unusual flight path (emergency squawk)
    """

    BASE_URL = "https://opensky-network.org/api"

    def fetch_aircraft(self, bounds: Dict) -> List[Dict]:
        """Fetch aircraft in bounding box."""
        url = (f"{self.BASE_URL}/states/all?"
               f"lamin={bounds['lat_min']}&lomin={bounds['lon_min']}&"
               f"lamax={bounds['lat_max']}&lomax={bounds['lon_max']}")

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                states = data.get("states", [])

                aircraft = []
                for s in states:
                    if s[5] is not None and s[6] is not None:  # Has position
                        aircraft.append({
                            "icao24": s[0],
                            "callsign": (s[1] or "").strip(),
                            "country": s[2],
                            "longitude": s[5],
                            "latitude": s[6],
                            "altitude": s[7] or s[13],  # baro or geo
                            "on_ground": s[8],
                            "velocity": s[9],
                            "heading": s[10],
                            "vertical_rate": s[11],
                            "squawk": s[14],
                        })
                return aircraft
        except Exception as e:
            print(f"  [OpenSky] Error: {e}")
            return []


class SpaceWeatherClient:
    """
    NOAA Space Weather Prediction Center.

    Updates: Varies (alerts as they happen)

    Outliers to detect:
    - Solar flare alerts
    - Geomagnetic storm warnings
    - High Kp index (>5)
    """

    ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"
    KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    def fetch_alerts(self) -> List[Dict]:
        """Fetch space weather alerts."""
        try:
            with urllib.request.urlopen(self.ALERTS_URL, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [SpaceWeather] Alerts error: {e}")
            return []

    def fetch_kp_index(self) -> List[Dict]:
        """Fetch planetary K index (geomagnetic activity)."""
        try:
            with urllib.request.urlopen(self.KP_URL, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                # Skip header row
                return [{"time": row[0], "kp": float(row[1])} for row in data[1:] if len(row) > 1]
        except Exception as e:
            print(f"  [SpaceWeather] Kp error: {e}")
            return []


class USGSWaterClient:
    """
    USGS Water Services - Stream flow, gage height.

    Updates: Every 15 minutes

    Outliers to detect:
    - Flash flood (rapid rise)
    - Drought (extreme low)
    - Dam release (sudden change)

    SF Bay Area stations:
    - 11162500: Pescadero Creek
    - 11169025: Guadalupe River
    - 11180500: Alameda Creek
    """

    BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

    STATIONS = {
        "11162500": "Pescadero Creek",
        "11169025": "Guadalupe River",
        "11180500": "Alameda Creek",
    }

    def fetch_streamflow(self, station_id: str) -> List[Dict]:
        """Fetch recent streamflow data."""
        params = {
            "format": "json",
            "sites": station_id,
            "parameterCd": "00060",  # Discharge (cfs)
            "period": "PT6H",  # Last 6 hours
        }

        url = self.BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())

                readings = []
                for ts in data.get("value", {}).get("timeSeries", []):
                    for val in ts.get("values", [{}])[0].get("value", []):
                        readings.append({
                            "station": station_id,
                            "timestamp": val.get("dateTime"),
                            "discharge_cfs": float(val.get("value", 0)),
                        })
                return readings
        except Exception as e:
            print(f"  [USGSWater] Error: {e}")
            return []


class AirQualityClient:
    """
    Open-Meteo Air Quality API.

    Updates: Hourly
    Free: Yes, no API key

    Outliers to detect:
    - AQI spikes (>150 unhealthy)
    - PM2.5 anomalies
    - Wildfire smoke events
    """

    BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def fetch_aqi(self, lat: float, lon: float) -> Dict:
        """Fetch air quality data."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi",
            "timezone": "America/Los_Angeles",
        }

        url = self.BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [AirQuality] Error: {e}")
            return {}


class NDBCBuoyClient:
    """
    NDBC Buoy data (expanded metrics).

    Updates: Every 10 minutes

    Full metrics available:
    - WDIR: Wind direction (degrees)
    - WSPD: Wind speed (m/s)
    - GST: Wind gust (m/s)
    - WVHT: Significant wave height (m)
    - DPD: Dominant wave period (sec)
    - APD: Average wave period (sec)
    - MWD: Wave direction (degrees)
    - PRES: Sea level pressure (hPa)
    - ATMP: Air temperature (°C)
    - WTMP: Water temperature (°C)
    - DEWP: Dewpoint (°C)
    - VIS: Visibility (nmi)
    - PTDY: Pressure tendency (hPa)
    - TIDE: Tide (ft)
    """

    BASE_URL = "https://www.ndbc.noaa.gov/data/realtime2"

    STATIONS = {
        "46026": {"name": "San Francisco", "lat": 37.750, "lon": -122.838},
        "46012": {"name": "Half Moon Bay", "lat": 37.356, "lon": -122.881},
        "46042": {"name": "Monterey", "lat": 36.789, "lon": -122.469},
        "46013": {"name": "Bodega Bay", "lat": 38.253, "lon": -123.303},
        "46214": {"name": "Point Reyes", "lat": 37.946, "lon": -123.469},
    }

    def fetch_all_metrics(self, station_id: str, hours: int = 6) -> List[Dict]:
        """Fetch all available metrics from buoy."""
        url = f"{self.BASE_URL}/{station_id}.txt"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                text = resp.read().decode()
        except Exception as e:
            print(f"  [NDBC] Error fetching {station_id}: {e}")
            return []

        readings = []
        lines = [l for l in text.strip().split('\n') if not l.startswith('#')]

        for line in lines[:hours * 6]:  # ~10 min intervals
            parts = line.split()
            if len(parts) < 17:
                continue

            def parse(s):
                return None if s == 'MM' else float(s)

            try:
                readings.append({
                    "station": station_id,
                    "timestamp": datetime(
                        int(parts[0]), int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4])
                    ),
                    "wind_dir": parse(parts[5]),
                    "wind_speed": parse(parts[6]),
                    "wind_gust": parse(parts[7]),
                    "wave_height": parse(parts[8]),
                    "wave_period_dominant": parse(parts[9]),
                    "wave_period_avg": parse(parts[10]),
                    "wave_dir": int(parts[11]) if parts[11] != 'MM' else None,
                    "pressure": parse(parts[12]),
                    "air_temp": parse(parts[13]),
                    "water_temp": parse(parts[14]),
                    "dewpoint": parse(parts[15]),
                    "visibility": parse(parts[16]) if len(parts) > 16 else None,
                })
            except (ValueError, IndexError):
                continue

        return readings


class USGSEarthquakeClient:
    """USGS Earthquake API (same as before but cleaner)."""

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def fetch_recent(self, lat: float, lon: float, radius_km: int = 200,
                     min_mag: float = 1.0, hours: int = 24) -> List[Dict]:
        """Fetch recent earthquakes."""
        start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

        params = {
            "format": "geojson",
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": radius_km,
            "minmagnitude": min_mag,
            "starttime": start,
            "orderby": "time",
        }

        url = self.BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())

                quakes = []
                for f in data.get("features", []):
                    props = f.get("properties", {})
                    coords = f.get("geometry", {}).get("coordinates", [])

                    quakes.append({
                        "id": f.get("id"),
                        "magnitude": props.get("mag"),
                        "place": props.get("place"),
                        "time": datetime.fromtimestamp(props.get("time", 0) / 1000),
                        "depth": coords[2] if len(coords) > 2 else None,
                        "longitude": coords[0],
                        "latitude": coords[1],
                    })
                return quakes
        except Exception as e:
            print(f"  [USGS EQ] Error: {e}")
            return []


# =============================================================================
# MULTI-FEED AGGREGATOR
# =============================================================================

class MultiFeedAggregator:
    """
    Aggregates data from all feeds and detects outliers.

    This is the main entry point.
    """

    def __init__(self):
        # Clients
        self.opensky = OpenSkyClient()
        self.space = SpaceWeatherClient()
        self.water = USGSWaterClient()
        self.air = AirQualityClient()
        self.buoy = NDBCBuoyClient()
        self.earthquake = USGSEarthquakeClient()

        # Detector
        self.detector = UnifiedOutlierDetector(window_size=100)

        # Results
        self.outliers: List[Outlier] = []
        self.stats: Dict[str, int] = {}

    def fetch_all(self, bounds: Dict = None) -> Dict:
        """Fetch from all feeds and detect outliers."""
        bounds = bounds or SF_BAY_BOUNDS
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2

        results = {
            "timestamp": datetime.now().isoformat(),
            "feeds": {},
            "outliers": [],
            "stats": {},
        }

        print(f"Fetching from all feeds ({datetime.now().strftime('%H:%M:%S')})...")

        # 1. Aircraft
        print("  [1/6] OpenSky Aircraft...")
        aircraft = self.opensky.fetch_aircraft(bounds)
        results["feeds"]["aircraft"] = len(aircraft)
        self.stats["aircraft_total"] = len(aircraft)

        for ac in aircraft:
            if ac["altitude"] and not ac["on_ground"]:
                self.detector.add("opensky", "altitude", ac["altitude"])
                outlier = self.detector.check(
                    "opensky", "altitude", ac["altitude"],
                    datetime.now(),
                    location=(ac["latitude"], ac["longitude"]),
                    metadata={"callsign": ac["callsign"], "squawk": ac["squawk"]},
                    z_threshold=3.0  # Aircraft vary a lot
                )
                if outlier:
                    results["outliers"].append(outlier)

        # 2. Buoys (all metrics)
        print("  [2/6] NDBC Buoys (all metrics)...")
        buoy_count = 0
        for station_id in ["46026", "46014", "46013"]:
            readings = self.buoy.fetch_all_metrics(station_id, hours=6)
            buoy_count += len(readings)

            for r in readings:
                ts = r["timestamp"]
                loc = (self.buoy.STATIONS.get(station_id, {}).get("lat", 0),
                       self.buoy.STATIONS.get(station_id, {}).get("lon", 0))

                # Check each metric
                metrics = [
                    ("wave_height", r.get("wave_height"), 2.5),
                    ("water_temp", r.get("water_temp"), 2.5),
                    ("wind_speed", r.get("wind_speed"), 2.5),
                    ("wind_gust", r.get("wind_gust"), 3.0),
                    ("pressure", r.get("pressure"), 3.0),
                    ("wave_period_dominant", r.get("wave_period_dominant"), 2.5),
                ]

                for metric, value, threshold in metrics:
                    if value is not None:
                        self.detector.add(f"ndbc_{station_id}", metric, value)
                        outlier = self.detector.check(
                            f"ndbc_{station_id}", metric, value, ts,
                            location=loc,
                            z_threshold=threshold
                        )
                        if outlier:
                            results["outliers"].append(outlier)

        results["feeds"]["buoy_readings"] = buoy_count

        # 3. Earthquakes
        print("  [3/6] USGS Earthquakes...")
        quakes = self.earthquake.fetch_recent(center_lat, center_lon, hours=24)
        results["feeds"]["earthquakes"] = len(quakes)

        for q in quakes:
            if q["magnitude"]:
                self.detector.add("usgs_eq", "magnitude", q["magnitude"])
                outlier = self.detector.check(
                    "usgs_eq", "magnitude", q["magnitude"],
                    q["time"],
                    location=(q["latitude"], q["longitude"]),
                    metadata={"place": q["place"], "depth": q["depth"]},
                    z_threshold=2.0  # Earthquakes are already rare events
                )
                if outlier:
                    results["outliers"].append(outlier)

        # 4. Space Weather
        print("  [4/6] NOAA Space Weather...")
        kp_data = self.space.fetch_kp_index()
        results["feeds"]["space_weather"] = len(kp_data)

        for kp in kp_data[-24:]:  # Last 24 readings
            self.detector.add("space", "kp_index", kp["kp"])
            outlier = self.detector.check(
                "space", "kp_index", kp["kp"],
                datetime.now(),
                metadata={"time_str": kp["time"]},
                z_threshold=2.0
            )
            if outlier:
                results["outliers"].append(outlier)

        # 5. Water (stream flow)
        print("  [5/6] USGS Water...")
        water_count = 0
        for station_id, name in self.water.STATIONS.items():
            readings = self.water.fetch_streamflow(station_id)
            water_count += len(readings)

            for r in readings:
                if r["discharge_cfs"]:
                    self.detector.add(f"usgs_water_{station_id}", "discharge", r["discharge_cfs"])
                    outlier = self.detector.check(
                        f"usgs_water_{station_id}", "discharge", r["discharge_cfs"],
                        datetime.now(),
                        metadata={"station_name": name},
                        z_threshold=3.0
                    )
                    if outlier:
                        results["outliers"].append(outlier)

        results["feeds"]["water_readings"] = water_count

        # 6. Air Quality
        print("  [6/6] Air Quality...")
        aqi_data = self.air.fetch_aqi(center_lat, center_lon)
        hourly = aqi_data.get("hourly", {})
        aqi_values = hourly.get("us_aqi", [])
        pm25_values = hourly.get("pm2_5", [])

        results["feeds"]["air_quality_hours"] = len(aqi_values)

        if aqi_values:
            current_aqi = aqi_values[0]
            self.detector.add("air", "aqi", current_aqi)
            outlier = self.detector.check(
                "air", "aqi", current_aqi,
                datetime.now(),
                location=(center_lat, center_lon),
                z_threshold=2.0
            )
            if outlier:
                results["outliers"].append(outlier)

        if pm25_values and pm25_values[0]:
            self.detector.add("air", "pm25", pm25_values[0])
            outlier = self.detector.check(
                "air", "pm25", pm25_values[0],
                datetime.now(),
                location=(center_lat, center_lon),
                z_threshold=2.5
            )
            if outlier:
                results["outliers"].append(outlier)

        # Store outliers
        self.outliers.extend(results["outliers"])

        # Calculate stats
        results["stats"] = {
            "total_data_points": sum(results["feeds"].values()),
            "outliers_detected": len(results["outliers"]),
            "outlier_rate": len(results["outliers"]) / max(sum(results["feeds"].values()), 1),
        }

        return results

    def summarize(self, results: Dict) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            "MULTI-FEED OUTLIER DETECTION SUMMARY",
            f"Timestamp: {results['timestamp']}",
            "=" * 70,
            "",
            "DATA FEEDS:",
        ]

        for feed, count in results["feeds"].items():
            lines.append(f"  {feed}: {count} data points")

        lines.extend([
            "",
            f"TOTAL DATA POINTS: {results['stats']['total_data_points']}",
            f"OUTLIERS DETECTED: {results['stats']['outliers_detected']}",
            f"OUTLIER RATE: {results['stats']['outlier_rate']:.2%}",
            "",
        ])

        if results["outliers"]:
            lines.append("OUTLIERS:")
            for o in results["outliers"][:10]:
                lines.append(f"  [{o.severity}] {o.feed}/{o.metric}: {o.value:.2f} (z={o.z_score:.1f})")
        else:
            lines.append("No outliers detected (all data within normal ranges)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# SCHEDULER
# =============================================================================

def calculate_run_schedule() -> Dict[str, int]:
    """
    Calculate optimal run schedule based on API update frequencies.

    Returns dict of feed -> seconds between runs
    """
    return {
        "fast": 60,      # Earthquakes, space weather (check every minute)
        "medium": 600,   # Buoys, aircraft (every 10 minutes)
        "slow": 3600,    # Air quality, marine forecast (hourly)
    }


def continuous_monitor(duration_minutes: int = 10):
    """
    Run continuous monitoring for specified duration.

    This shows how the system would run in production.
    """
    print("=" * 70)
    print("CONTINUOUS MONITORING MODE")
    print(f"Duration: {duration_minutes} minutes")
    print("=" * 70)
    print()

    aggregator = MultiFeedAggregator()
    start_time = datetime.now()
    run_count = 0

    while (datetime.now() - start_time).seconds < duration_minutes * 60:
        run_count += 1
        print(f"\n--- Run #{run_count} ---")

        results = aggregator.fetch_all()

        print(f"  Data points: {results['stats']['total_data_points']}")
        print(f"  Outliers: {results['stats']['outliers_detected']}")

        if results["outliers"]:
            print("  Outliers found:")
            for o in results["outliers"][:3]:
                print(f"    [{o.severity}] {o.feed}/{o.metric}: {o.value:.2f}")

        # Wait before next run
        wait_time = 60  # 1 minute between runs for demo
        print(f"\n  Waiting {wait_time}s until next run...")
        time.sleep(wait_time)

    print("\n" + "=" * 70)
    print("MONITORING COMPLETE")
    print(f"Total runs: {run_count}")
    print(f"Total outliers detected: {len(aggregator.outliers)}")
    print("=" * 70)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the multi-feed system."""
    print("=" * 70)
    print("MULTI-FEED OUTLIER DETECTION - DEMO")
    print("=" * 70)
    print()
    print("APIs being queried:")
    print("  1. OpenSky Network (aircraft)")
    print("  2. NDBC Buoys (wave, temp, wind, pressure)")
    print("  3. USGS Earthquakes")
    print("  4. NOAA Space Weather (Kp index)")
    print("  5. USGS Water (stream flow)")
    print("  6. Open-Meteo Air Quality")
    print()

    aggregator = MultiFeedAggregator()
    results = aggregator.fetch_all()

    print()
    print(aggregator.summarize(results))

    print()
    print("UPDATE FREQUENCIES:")
    for feed, interval in UPDATE_INTERVALS.items():
        print(f"  {feed}: every {interval}s ({interval/60:.1f} min)")

    print()
    print("RECOMMENDED CRON SCHEDULE:")
    print("  */1 * * * * - Earthquakes, space weather (every minute)")
    print("  */10 * * * * - Buoys, aircraft (every 10 minutes)")
    print("  0 * * * * - Air quality, forecasts (hourly)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demo()
