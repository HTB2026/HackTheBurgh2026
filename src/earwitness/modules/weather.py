"""Weather witness: what the sky was really doing there and then.

If the place (GPS, or a verified geolocation) and the time (metadata that is
not a copy stamp, or an ENF date) are known, fetch the recorded weather for
that hour from Open-Meteo's historical archive and compare it with what the
AI analyst sees in the frames. Rain in the record and a dry, sunny video (or
the reverse) means the claimed time or place is wrong.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import httpx

from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}  # fmt: skip
_WET_WORDS = ("rain", "wet", "drizzle", "puddle", "umbrella", "snow", "shower", "storm")
_DRY_WORDS = ("dry", "sunny", "clear sky", "clear, ", "bright sun")


async def _hourly(lat: float, lon: float, when: datetime) -> dict[str, Any] | None:
    day = when.astimezone(timezone.utc).date()
    params = {
        "latitude": lat, "longitude": lon, "start_date": day.isoformat(), "end_date": day.isoformat(),
        "hourly": "temperature_2m,precipitation,cloud_cover,weather_code,wind_speed_10m,is_day",
        "timezone": "UTC",
    }  # fmt: skip
    recent = datetime.now(timezone.utc).date() - day < timedelta(days=6)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        if recent
        else "https://archive-api.open-meteo.com/v1/archive"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()["hourly"]
    except httpx.HTTPError, KeyError, ValueError:
        return None

    hour = when.astimezone(timezone.utc).hour

    if hour >= len(data.get("time", [])):
        return None

    return {key: values[hour] for key, values in data.items()} | {"source": url}


class Weather:
    name = "weather"
    title = "WEATHER WITNESS"
    description = "Real recorded weather at the claimed place and time, checked against what the video shows."
    depends = ["provenance", "enf", "geo", "vision"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        provenance = await ctx.wait_for("provenance") or {}
        enf = await ctx.wait_for("enf") or {}
        geo = await ctx.wait_for("geo") or {}
        place, place_source = None, None

        if provenance.get("location"):
            place, place_source = provenance["location"], "the file's GPS"
        elif geo.get("verified"):
            place, place_source = (
                {"lat": geo["latitude"], "lon": geo["longitude"]},
                "the verified geolocation",
            )

        when, when_source = None, None

        if enf.get("confident") and enf.get("dated_start"):
            when, when_source = enf["dated_start"], "the power-grid dating"
        elif provenance.get("start_guess"):
            when, when_source = provenance["start_guess"], "the file's own timestamp"

        if isinstance(when, str):
            when = datetime.fromisoformat(when)

        if place is None or when is None:
            raise Skip("needs both a place and a trustworthy time")

        out.progress(0.3, "fetching the weather record")
        record = await _hourly(float(place["lat"]), float(place["lon"]), when)

        if record is None:
            raise Skip("no weather record for that place and hour")

        sky = _CODES.get(int(record.get("weather_code") or 0), "unknown")
        summary = (
            f"{record['temperature_2m']:.0f}°C, {sky}, {record['precipitation']:.1f} mm of rain that hour, "
            f"{record['cloud_cover']:.0f}% cloud, wind {record['wind_speed_10m']:.0f} km/h, "
            f"{'daylight' if record.get('is_day') else 'dark'}"
        )
        out.artifact(
            "record",
            "table",
            "Recorded weather for that place and hour",
            data={
                "columns": ["measure", "value"],
                "rows": [[k, v] for k, v in record.items()],
            },
        )
        vision = await ctx.wait_for("vision") or {}
        seen = f"{vision.get('weather', '')} {vision.get('scene', '')}".lower()
        wet_record = float(record.get("precipitation") or 0) >= 0.5 or sky.endswith(
            ("rain", "drizzle", "showers")
        )
        dry_record = (
            float(record.get("precipitation") or 0) == 0
            and int(record.get("weather_code") or 0) <= 2
        )
        wet_seen = any(w in seen for w in _WET_WORDS)
        dry_seen = any(w in seen for w in _DRY_WORDS) and not wet_seen
        contradiction = (wet_record and dry_seen) or (dry_record and wet_seen)
        detail = (
            f"At {when:%Y-%m-%d %H:00} UTC at that spot (place from {place_source}, time from "
            f"{when_source}) the weather record says: {summary}."
        )

        if seen.strip():
            detail += f" The AI analyst sees: {vision.get('weather') or 'no weather described'}."

        out.finding(
            "warning" if contradiction else "info",
            "Weather contradicts the claimed time or place"
            if contradiction
            else f"Weather there and then: {sky}, {record['temperature_2m']:.0f}°C",
            detail
            + (
                " These do not fit together: either the time, the place, or the footage is not what it claims."
                if contradiction
                else ""
            ),
            tags=["weather", "time", "location"]
            + (["inconsistency"] if contradiction else []),
            data=record
            | {"sky": sky, "place_source": place_source, "time_source": when_source},
        )
        ctx.publish(self.name, record | {"sky": sky, "contradiction": contradiction})
        out.progress(1.0)
