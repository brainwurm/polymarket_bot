"""
weather_edge.py — Weather Model vs Polymarket Mispricing Agent
==============================================================
Inspired by @Vvtentt101's thread on predictable weather cities.

Core idea:
  Free numerical weather models (GFS, ECMWF via Open-Meteo) have genuine
  predictive skill up to 10-12 days out in flat-terrain, single-climate cities.
  When Polymarket crowd pricing disagrees with model consensus, that's edge.

Data sources (all free, no API key required):
  - Open-Meteo API  — global ECMWF + GFS ensemble forecasts
  - api.weather.gov — NOAA official US forecasts (backup)

Cities we target (from the tweet — flat terrain, model agreement, tradeable):
  - Dallas, TX       (GFS v16, flat plains, +12d edge window Oct-May)
  - Atlanta, GA      (NAM, subtropical, +10d year-round)
  - Seoul, South Korea (ECMWF, winter dominant, +10d Nov-Feb)
  - Shanghai, China  (ECMWF, +11d Dec-Feb and Jun-Aug)
  - Kansas City, MO  (flat, great plains, highly predictable)
  - Phoenix, AZ      (desert, extremely stable, easiest to predict)

Usage (standalone test):
    python3 weather_edge.py

Usage (as module in orchestrator):
    from weather_edge import get_weather_signals, WeatherSignal
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger("weather-edge")

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
NOAA_BASE       = "https://api.weather.gov"

# ── Target cities with metadata from the tweet ────────────────────────────────
TARGET_CITIES = [
    {
        "name":         "Dallas",
        "state":        "TX",
        "country":      "US",
        "latitude":     32.7767,
        "longitude":    -96.7970,
        "model":        "GFS",
        "edge_days":    12,        # forecast horizon with reliable skill
        "best_months":  [10, 11, 12, 1, 2, 3, 4, 5],   # Oct-May
        "keywords":     ["dallas", "dfw", "fort worth", "texas"],
        "terrain":      "flat_plains",
    },
    {
        "name":         "Atlanta",
        "state":        "GA",
        "country":      "US",
        "latitude":     33.7490,
        "longitude":    -84.3880,
        "model":        "NAM",
        "edge_days":    10,
        "best_months":  list(range(1, 13)),              # year-round
        "keywords":     ["atlanta", "georgia", "atl"],
        "terrain":      "mild_subtropical",
    },
    {
        "name":         "Seoul",
        "country":      "South Korea",
        "latitude":     37.5665,
        "longitude":    126.9780,
        "model":        "ECMWF",
        "edge_days":    10,
        "best_months":  [11, 12, 1, 2],                 # Nov-Feb winter edge
        "keywords":     ["seoul", "korea", "incheon"],
        "terrain":      "continental_flat",
    },
    {
        "name":         "Shanghai",
        "country":      "China",
        "latitude":     31.2304,
        "longitude":    121.4737,
        "model":        "ECMWF",
        "edge_days":    11,
        "best_months":  [12, 1, 2, 6, 7, 8],            # Dec-Feb, Jun-Aug
        "keywords":     ["shanghai", "china"],
        "terrain":      "flat_coastal",
    },
    {
        "name":         "Phoenix",
        "state":        "AZ",
        "country":      "US",
        "latitude":     33.4484,
        "longitude":    -112.0740,
        "model":        "GFS",
        "edge_days":    14,                              # desert = most predictable
        "best_months":  list(range(1, 13)),
        "keywords":     ["phoenix", "arizona", "az", "scottsdale", "tempe"],
        "terrain":      "desert",
    },
    {
        "name":         "Kansas City",
        "state":        "MO",
        "country":      "US",
        "latitude":     39.0997,
        "longitude":    -94.5786,
        "model":        "GFS",
        "edge_days":    12,
        "best_months":  [10, 11, 12, 1, 2, 3, 4, 5],
        "keywords":     ["kansas city", "kc", "missouri"],
        "terrain":      "flat_plains",
    },
]

# ── Weather forecast fetcher ──────────────────────────────────────────────────
@dataclass
class CityForecast:
    city_name:    str
    latitude:     float
    longitude:    float
    model:        str
    forecast_days: int
    daily_max_f:  list[float]    # max temps in Fahrenheit per day
    daily_min_f:  list[float]    # min temps in Fahrenheit per day
    daily_precip: list[float]    # precipitation probability per day (0-100)
    fetched_at:   float = field(default_factory=time.time)

    @property
    def today_max_f(self) -> float:
        return self.daily_max_f[0] if self.daily_max_f else 0.0

    @property
    def today_min_f(self) -> float:
        return self.daily_min_f[0] if self.daily_min_f else 0.0

    def max_f_on_day(self, day_offset: int) -> Optional[float]:
        """Get forecasted max temp for N days from now (0=today)."""
        if day_offset < len(self.daily_max_f):
            return self.daily_max_f[day_offset]
        return None

    def min_f_on_day(self, day_offset: int) -> Optional[float]:
        if day_offset < len(self.daily_min_f):
            return self.daily_min_f[day_offset]
        return None

    def precip_prob_on_day(self, day_offset: int) -> Optional[float]:
        if day_offset < len(self.daily_precip):
            return self.daily_precip[day_offset]
        return None

def celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32

def fetch_forecast(city: dict, forecast_days: int = 14) -> Optional[CityForecast]:
    """
    Fetch multi-day weather forecast from Open-Meteo.
    Uses ECMWF ensemble for international cities, GFS for US.
    Free, no API key required.
    """
    # Choose model based on city metadata
    model_param = "ecmwf_ifs025" if city["model"] == "ECMWF" else "gfs_seamless"

    try:
        resp = requests.get(
            OPEN_METEO_BASE,
            params={
                "latitude":              city["latitude"],
                "longitude":             city["longitude"],
                "daily":                 "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit":      "celsius",
                "forecast_days":         forecast_days,
                "timezone":              "auto",
                "models":                model_param,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Open-Meteo fetch failed for {city['name']}: {e}")
        return None

    daily = data.get("daily", {})
    max_c = daily.get("temperature_2m_max", [])
    min_c = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_probability_max", [])

    if not max_c:
        return None

    return CityForecast(
        city_name=    city["name"],
        latitude=     city["latitude"],
        longitude=    city["longitude"],
        model=        city["model"],
        forecast_days=forecast_days,
        daily_max_f=  [celsius_to_fahrenheit(t) for t in max_c],
        daily_min_f=  [celsius_to_fahrenheit(t) for t in min_c],
        daily_precip= [float(p) if p is not None else 0.0 for p in precip],
    )

# ── Weather signal generator ──────────────────────────────────────────────────
@dataclass
class WeatherSignal:
    city_name:      str
    city_keywords:  list[str]
    question_type:  str           # "temp_above", "temp_below", "precip"
    threshold:      float         # the temperature or precip threshold
    day_offset:     int           # how many days from now
    model_prob:     float         # model's estimated probability (0-1)
    model_name:     str
    edge_days:      int           # city's forecast reliability horizon
    confidence:     float         # our confidence in the signal
    rationale:      str
    timestamp:      float = field(default_factory=time.time)

    @property
    def is_within_edge_window(self) -> bool:
        return self.day_offset <= self.edge_days

    @property
    def in_season(self) -> bool:
        """Check if current month is in the city's best edge months."""
        current_month = int(time.strftime("%m"))
        city = next((c for c in TARGET_CITIES if c["name"] == self.city_name), None)
        if city is None:
            return True
        return current_month in city.get("best_months", list(range(1, 13)))

def generate_temp_signals(forecast: CityForecast, city: dict) -> list[WeatherSignal]:
    """
    Generate temperature threshold signals.
    For each day in the edge window, compute model probability of
    exceeding common Polymarket temperature thresholds.

    Polymarket weather markets typically ask things like:
    "Will the high temperature in Dallas exceed 100°F on July 15?"
    "Will Atlanta's low drop below 32°F in January?"
    """
    signals = []
    edge_days = city["edge_days"]

    # Common temperature thresholds Polymarket uses by city type
    terrain = city.get("terrain", "")
    if terrain == "desert":
        # Phoenix: heat records, 110°F+ days
        thresholds_max = [95, 100, 105, 110, 115]
        thresholds_min = [32, 40, 50]
    elif terrain == "flat_plains":
        # Dallas, Kansas City
        thresholds_max = [85, 90, 95, 100, 105]
        thresholds_min = [20, 25, 32, 40]
    elif terrain == "mild_subtropical":
        # Atlanta
        thresholds_max = [85, 90, 95, 100]
        thresholds_min = [25, 32, 40]
    elif terrain == "continental_flat":
        # Seoul
        thresholds_max = [85, 90, 95]
        thresholds_min = [14, 23, 32]
    else:
        thresholds_max = [85, 90, 95, 100]
        thresholds_min = [25, 32, 40]

    for day_offset in range(min(edge_days, forecast.forecast_days)):
        max_f = forecast.max_f_on_day(day_offset)
        min_f = forecast.min_f_on_day(day_offset)
        if max_f is None or min_f is None:
            continue

        # Temperature model uncertainty grows with forecast horizon
        # Based on typical NWP skill scores for flat terrain cities
        sigma_f = 2.5 + day_offset * 0.4    # standard deviation in °F

        # Generate signals for each threshold
        for threshold in thresholds_max:
            # Model probability of exceeding threshold using normal distribution
            import math
            z = (threshold - max_f) / sigma_f
            # P(X > threshold) = 1 - CDF(z)
            model_prob = 1 - _normal_cdf(z)

            # Only signal when probability is meaningfully different from 0.5
            # (near 0.5 = uncertain, not tradeable)
            if model_prob < 0.15 or model_prob > 0.85:
                # High conviction signal
                conf = _compute_confidence(day_offset, edge_days, model_prob, city)
                signals.append(WeatherSignal(
                    city_name=     forecast.city_name,
                    city_keywords= city["keywords"],
                    question_type= "temp_above",
                    threshold=     threshold,
                    day_offset=    day_offset,
                    model_prob=    round(model_prob, 4),
                    model_name=    forecast.model,
                    edge_days=     edge_days,
                    confidence=    conf,
                    rationale=(
                        f"{forecast.model} forecasts {forecast.city_name} max "
                        f"{max_f:.1f}°F on day+{day_offset} "
                        f"(±{sigma_f:.1f}°F) → P(>{threshold}°F)={model_prob:.1%}"
                    ),
                ))

        for threshold in thresholds_min:
            z = (threshold - min_f) / sigma_f
            model_prob = _normal_cdf(z)    # P(X < threshold)

            if model_prob < 0.15 or model_prob > 0.85:
                conf = _compute_confidence(day_offset, edge_days, model_prob, city)
                signals.append(WeatherSignal(
                    city_name=     forecast.city_name,
                    city_keywords= city["keywords"],
                    question_type= "temp_below",
                    threshold=     threshold,
                    day_offset=    day_offset,
                    model_prob=    round(model_prob, 4),
                    model_name=    forecast.model,
                    edge_days=     edge_days,
                    confidence=    conf,
                    rationale=(
                        f"{forecast.model} forecasts {forecast.city_name} min "
                        f"{min_f:.1f}°F on day+{day_offset} "
                        f"(±{sigma_f:.1f}°F) → P(<{threshold}°F)={model_prob:.1%}"
                    ),
                ))

    return signals

def generate_precip_signals(forecast: CityForecast, city: dict) -> list[WeatherSignal]:
    """
    Generate precipitation probability signals.
    Model precip probability is already 0-100, convert to 0-1.
    """
    signals = []
    edge_days = min(city["edge_days"], 7)   # precip harder to forecast, use shorter window

    for day_offset in range(min(edge_days, forecast.forecast_days)):
        precip_pct = forecast.precip_prob_on_day(day_offset)
        if precip_pct is None:
            continue

        model_prob = precip_pct / 100.0

        # Only trade when model is confident (>80% or <20%)
        if model_prob < 0.20 or model_prob > 0.80:
            conf = _compute_confidence(day_offset, edge_days, model_prob, city)
            signals.append(WeatherSignal(
                city_name=     forecast.city_name,
                city_keywords= city["keywords"],
                question_type= "precip",
                threshold=     0.0,
                day_offset=    day_offset,
                model_prob=    round(model_prob, 4),
                model_name=    forecast.model,
                edge_days=     edge_days,
                confidence=    conf,
                rationale=(
                    f"{forecast.model} gives {model_prob:.0%} precipitation "
                    f"probability for {forecast.city_name} on day+{day_offset}"
                ),
            ))

    return signals

def _normal_cdf(z: float) -> float:
    """Standard normal CDF — no scipy needed, pure Python."""
    import math
    return (1.0 + math.erf(z / math.sqrt(2))) / 2.0

def _compute_confidence(
    day_offset: int,
    edge_days: int,
    model_prob: float,
    city: dict,
) -> float:
    """
    Confidence decays with forecast horizon and increases with model conviction.
    Max confidence for day+0 with 95%+ model probability.
    Floor at MIN_CONFIDENCE threshold.
    """
    current_month = int(time.strftime("%m"))
    in_season = current_month in city.get("best_months", list(range(1, 13)))

    # Base: how far into the edge window are we? Closer = more confident
    horizon_factor = 1.0 - (day_offset / edge_days) * 0.35

    # Model conviction: farther from 0.5 = higher confidence
    conviction = abs(model_prob - 0.5) * 2   # 0 to 1

    # Season bonus: cities have higher skill in their optimal months
    season_bonus = 0.05 if in_season else 0.0

    # Terrain bonus: flat/desert cities are more predictable
    terrain_bonus = {
        "desert":          0.08,
        "flat_plains":     0.05,
        "mild_subtropical":0.03,
        "continental_flat":0.03,
        "flat_coastal":    0.02,
    }.get(city.get("terrain", ""), 0.0)

    confidence = (
        0.60 * horizon_factor +
        0.08 * conviction +
        season_bonus +
        terrain_bonus
    )

    return round(min(0.88, max(0.60, confidence)), 3)

# ── Match weather signals to Polymarket markets ───────────────────────────────
def match_weather_market(
    weather_sig: WeatherSignal,
    market: dict,
) -> Optional[tuple[str, float, float, str]]:
    """
    Try to match a weather signal to a Polymarket market question.

    Returns (outcome, current_price, fair_value, rationale) or None.

    Polymarket weather question formats we look for:
    - "Will the high temperature in Dallas exceed 100°F on [date]?"
    - "Will it rain in Atlanta on [date]?"
    - "Will Dallas reach 95°F in [month]?"
    """
    question = market.get("question", "").lower()

    # Check if this market is about our city
    city_match = any(kw in question for kw in weather_sig.city_keywords)
    if not city_match:
        return None

    # Check question type matches signal type
    yes_price = float((market.get("outcomePrices") or ["0.5"])[0])

    if weather_sig.question_type == "temp_above":
        # Look for temperature threshold questions
        threshold_str = str(int(weather_sig.threshold))
        if threshold_str not in question:
            return None
        if "exceed" not in question and "above" not in question and "reach" not in question and "hit" not in question:
            return None
        outcome    = "YES" if weather_sig.model_prob > 0.5 else "NO"
        fair_value = weather_sig.model_prob if outcome == "YES" else (1 - weather_sig.model_prob)

    elif weather_sig.question_type == "temp_below":
        threshold_str = str(int(weather_sig.threshold))
        if threshold_str not in question:
            return None
        if "below" not in question and "under" not in question and "freeze" not in question and "frost" not in question:
            return None
        outcome    = "YES" if weather_sig.model_prob > 0.5 else "NO"
        fair_value = weather_sig.model_prob if outcome == "YES" else (1 - weather_sig.model_prob)

    elif weather_sig.question_type == "precip":
        if "rain" not in question and "precip" not in question and "snow" not in question and "storm" not in question:
            return None
        outcome    = "YES" if weather_sig.model_prob > 0.5 else "NO"
        fair_value = weather_sig.model_prob if outcome == "YES" else (1 - weather_sig.model_prob)

    else:
        return None

    # Calculate edge
    current_price = yes_price if outcome == "YES" else (1 - yes_price)
    edge = abs(fair_value - current_price)

    if edge < 0.06:
        return None

    return outcome, current_price, fair_value, weather_sig.rationale

# ── Main entry point called by orchestrator ───────────────────────────────────
def get_weather_signals() -> list[WeatherSignal]:
    """
    Fetch forecasts for all target cities and generate weather signals.
    Called by the weather-edge agent in orchestrator.py.
    No API key needed — Open-Meteo is completely free.
    """
    all_signals: list[WeatherSignal] = []

    for city in TARGET_CITIES:
        log.info(f"Fetching {city['model']} forecast for {city['name']}...")
        forecast = fetch_forecast(city, forecast_days=14)

        if forecast is None:
            log.warning(f"No forecast for {city['name']} — skipping")
            continue

        temp_signals  = generate_temp_signals(forecast, city)
        precip_signals = generate_precip_signals(forecast, city)

        city_signals = temp_signals + precip_signals
        log.info(
            f"  {city['name']}: today max={forecast.today_max_f:.1f}°F "
            f"min={forecast.today_min_f:.1f}°F | "
            f"{len(city_signals)} signals generated"
        )
        all_signals.extend(city_signals)
        time.sleep(0.3)   # polite rate limiting

    # Sort by confidence descending
    all_signals.sort(key=lambda s: s.confidence, reverse=True)
    return all_signals

# ── Standalone diagnostic ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("\n" + "="*70)
    print("  Weather Edge Diagnostic — Open-Meteo Free API")
    print("  Targeting: Dallas, Atlanta, Seoul, Shanghai, Phoenix, Kansas City")
    print("="*70)

    signals = get_weather_signals()

    print(f"\nTotal signals generated: {len(signals)}")
    print(f"\n{'City':<14} {'Type':<12} {'Thresh':>7} {'Day':>4} {'Model P':>8} {'Conf':>6}  Rationale")
    print("-" * 90)

    for s in signals[:20]:   # show top 20
        print(
            f"{s.city_name:<14} "
            f"{s.question_type:<12} "
            f"{s.threshold:>6.0f}°F "
            f"{s.day_offset:>4} "
            f"{s.model_prob:>8.1%} "
            f"{s.confidence:>6.1%}  "
            f"{s.rationale[:45]}..."
        )

    print("\n✓ Weather edge module ready — import get_weather_signals() in orchestrator.py")
    print("  No API key required. Uses Open-Meteo (ECMWF + GFS ensemble data).")
