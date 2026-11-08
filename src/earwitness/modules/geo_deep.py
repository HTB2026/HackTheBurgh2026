"""Extra geolocation methods for Deep-effort runs.

The standard pipeline (``geo.py``) proposes places and checks them against a
Wikipedia photo and one satellite tile, which works where there is a landmark
or text. The countryside has neither, so a Deep run adds checks that work on
fields, lanes, hills and skies:

* a clue inventory: what a GeoGuessr player reads in a rural scene (poles,
  road lines, walls, crops, soil, terrain), plus sun, shadows and stars;
* several independent proposals, clustered: agreement is evidence;
* ground-level photos near each candidate (Wikimedia Commons, Mapillary);
* OpenStreetMap: do the features seen (wind turbines, pylons, coast) exist there?
* terrain: does the skyline in the frame match the horizon rendered from the
  elevation model at the candidate?
* the sun: with a trustworthy time, its height above the horizon puts the
  camera on a ring around the point where the sun was overhead;
* stars: a plate-solved star field bounds the latitude;
* weather: rain in the frame and a dry record there (or the reverse) counts
  against a candidate.

Every function here either returns None (could not check) or a result that
states what it measured. Network calls are small functions so tests can patch
them; the pure maths (rings, horizons, clustering) is tested directly.
"""

import asyncio
import math
import os
from datetime import datetime
from datetime import timezone
from typing import Any

import cv2
import httpx
import numpy as np

EARTH_KM = 6371.0
_KM_PER_DEG = 111.195
_USER_AGENT = "Earwitness/0.2 (https://earwitness.ussr.pl; hackathon research demo)"

# ---- clue inventory ----------------------------------------------------------------

# OpenStreetMap features a frame can show, with the tags that mark them and
# how far away they could plausibly be while still visible (metres).
FEATURES: dict[str, tuple[list[str], int]] = {
    "wind_turbine": (['node["generator:source"="wind"]'], 12000),
    "power_line": (['way["power"="line"]'], 3000),
    "railway": (['way["railway"="rail"]'], 2000),
    "motorway": (['way["highway"="motorway"]'], 2000),
    "river": (['way["waterway"="river"]'], 3000),
    "lake": (['way["natural"="water"]', 'relation["natural"="water"]'], 4000),
    "coastline": (['way["natural"="coastline"]'], 10000),
    "church": (['node["amenity"="place_of_worship"]', 'way["amenity"="place_of_worship"]'], 3000),
    "lighthouse": (['node["man_made"="lighthouse"]', 'way["man_made"="lighthouse"]'], 15000),
    "castle": (['node["historic"="castle"]', 'way["historic"="castle"]'], 5000),
    "mountain_peak": (['node["natural"="peak"]'], 20000),
    "vineyard": (['way["landuse"="vineyard"]'], 2000),
    "forest": (['way["landuse"="forest"]', 'way["natural"="wood"]'], 1500),
    "golf_course": (['way["leisure"="golf_course"]'], 1500),
}  # fmt: skip
_FEATURE_WORDS = {
    "wind_turbine": "wind turbines", "power_line": "high-voltage power lines", "railway": "a railway",
    "motorway": "a motorway", "river": "a river", "lake": "a lake", "coastline": "the sea coast",
    "church": "a church", "lighthouse": "a lighthouse", "castle": "a castle",
    "mountain_peak": "mountain peaks", "vineyard": "vineyards", "forest": "woodland",
    "golf_course": "a golf course",
}  # fmt: skip

INVENTORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "scene_type", "driving_side", "road", "poles_and_wires", "roadside", "boundaries",
        "land_use", "vegetation", "soil_and_rock", "terrain", "horizon", "buildings",
        "vehicles", "osm_features", "lighting", "sun_visible", "sun_elevation_deg",
        "sun_basis", "shadow_direction", "hemisphere_hint", "hemisphere_reasoning",
        "stars_visible", "constellations", "camera_tilt_deg", "moon", "strongest_clues",
    ],
    "properties": {
        "scene_type": {"type": "string", "enum": ["urban", "suburban", "rural", "wilderness", "indoor", "water"]},
        "driving_side": {"type": "string", "enum": ["left", "right", "unknown"]},
        "road": {"type": "string", "description": "Surface, width, centre and edge line colours and pattern; 'none' if no road."},
        "poles_and_wires": {"type": "string", "description": "Utility pole material and shape, cross-arms, insulators, wires; 'none'."},
        "roadside": {"type": "string", "description": "Bollards (shape, colour, reflectors), guard rails, kerbs, sign shapes and colours even if unreadable."},
        "boundaries": {"type": "string", "description": "Fences, dry-stone walls, hedgerows, gates."},
        "land_use": {"type": "string", "description": "Fields and their shapes, crops, livestock, orchards, vineyards, forestry."},
        "vegetation": {"type": "string", "description": "Tree and plant species or families, season."},
        "soil_and_rock": {"type": "string", "description": "Soil colour, exposed rock type."},
        "terrain": {"type": "string", "description": "Flat, rolling, hilly, mountainous; valley, moor, plateau."},
        "horizon": {"type": "string", "enum": ["natural_hills", "mountains", "natural_flat", "sea", "buildings", "trees", "none"],
                    "description": "What forms the skyline across most of the frame."},
        "buildings": {"type": "string", "description": "Roof material and colour, wall material, style; 'none'."},
        "vehicles": {"type": "string", "description": "Makes, plate shapes and colours, farm machinery; 'none'."},
        "osm_features": {"type": "array", "items": {"type": "string", "enum": list(FEATURES)},
                         "description": "Only features clearly visible in the frames."},
        "lighting": {"type": "string", "enum": ["day", "twilight", "night", "indoor", "unknown"]},
        "sun_visible": {"type": "boolean", "description": "The sun itself, or sharp shadows that show where it is."},
        "sun_elevation_deg": {"type": ["number", "null"],
                              "description": "Sun height above the horizon in degrees, estimated from a vertical object and its shadow (elevation = atan(height / shadow length)) or from the sun's position in the frame; null if it cannot be judged."},
        "sun_basis": {"type": "string", "description": "What the elevation estimate rests on."},
        "shadow_direction": {"type": "string", "description": "Which way shadows fall in the frame; 'none'."},
        "hemisphere_hint": {"type": "string", "enum": ["north", "south", "unknown"]},
        "hemisphere_reasoning": {"type": "string"},
        "stars_visible": {"type": "boolean"},
        "constellations": {"type": "string", "description": "Constellations or bright stars you can identify; 'none'."},
        "camera_tilt_deg": {"type": ["number", "null"], "description": "How far above level the camera points (0 level, 90 straight up); null if unknown."},
        "moon": {"type": "string", "description": "Moon phase and position if visible; 'none'."},
        "strongest_clues": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
}  # fmt: skip

INVENTORY = (
    "You are a top GeoGuessr player and open-source geolocation analyst. Do NOT guess the place "
    "yet. Make an exact inventory of every location clue in these frames, especially the ones "
    "that work where there is no text and no landmark: utility poles, road line colours, "
    "bollards, guard rails, walls and hedges, field shapes, crops, livestock, soil colour, tree "
    "species, terrain, the skyline, roofs, the sun and shadows, stars. Describe only what you can "
    "see; write 'none' or 'unknown' rather than inventing. Estimate the sun's elevation only from "
    "a real measurement (a vertical object and its shadow, or the sun in frame)."
)

REFINE = (
    "You are a world-class geolocation analyst refining an earlier answer. The earlier passes put "
    "this footage near the region given below, but no exact spot was confirmed. You see the "
    "evidence frames, the clue inventory, the automatic checks, and satellite views of the region "
    "at two scales. Propose 3-5 EXACT spots inside or near that region (each with a radius of at "
    "most 3 km) where the frames could have been taken: match field patterns, road shapes, "
    "rivers, woods, slopes and settlements in the satellite views to the frames. Use real "
    "coordinates of real places; name the nearest village or road in 'place'."
)


def inventory_text(inv: dict[str, Any]) -> str:
    """The inventory as lines for the proposer's clue block."""
    keys = [
        ("scene_type", "Scene"), ("driving_side", "Driving side"), ("road", "Road"),
        ("poles_and_wires", "Poles and wires"), ("roadside", "Roadside furniture"),
        ("boundaries", "Field boundaries"), ("land_use", "Land use"), ("vegetation", "Vegetation"),
        ("soil_and_rock", "Soil and rock"), ("terrain", "Terrain"), ("horizon", "Skyline"),
        ("buildings", "Buildings"), ("vehicles", "Vehicles"), ("shadow_direction", "Shadows"),
        ("constellations", "Stars"), ("moon", "Moon"),
    ]  # fmt: skip
    lines = [
        f"{label}: {inv[key]}"
        for key, label in keys
        if str(inv.get(key) or "").strip().lower() not in ("", "none", "unknown")
    ]

    if inv.get("hemisphere_hint") in ("north", "south"):
        lines.append(
            f"Hemisphere hint: {inv['hemisphere_hint']} ({inv.get('hemisphere_reasoning', '')})"
        )

    if inv.get("strongest_clues"):
        lines.append("Strongest clues: " + "; ".join(inv["strongest_clues"]))

    return "\n".join(lines)


# ---- ensemble ----------------------------------------------------------------------


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )

    return 2 * EARTH_KM * math.asin(math.sqrt(min(1.0, h)))


def cluster(
    passes: list[list[dict[str, Any]]], within_km: float = 25.0, keep: int = 5
) -> list[dict[str, Any]]:
    """Merge candidates from independent proposal passes.

    Candidates within ``within_km`` of each other are one place. Each merged
    candidate keeps the most probable member's text and coordinates, its
    probability becomes the mean over all passes (a pass that missed it adds
    0), and ``votes`` counts the passes that proposed it.
    """
    groups: list[dict[str, Any]] = []
    flat = sorted(
        ((p, c) for p, cands in enumerate(passes) for c in cands),
        key=lambda pc: -float(pc[1].get("probability") or 0),
    )

    for pass_index, candidate in flat:
        here = (float(candidate["latitude"]), float(candidate["longitude"]))
        group = next(
            (
                g for g in groups
                if haversine_km(here, (g["lead"]["latitude"], g["lead"]["longitude"])) <= within_km
            ),
            None,
        )  # fmt: skip

        if group is None:
            groups.append({"lead": candidate, "passes": {pass_index}, "total": 0.0})
            group = groups[-1]

        group["passes"].add(pass_index)
        group["total"] += float(candidate.get("probability") or 0)

    total_passes = max(1, len(passes))
    merged = [
        g["lead"] | {
            "probability": round(g["total"] / total_passes, 3),
            "votes": len(g["passes"]),
            "passes": total_passes,
        }
        for g in groups
    ]  # fmt: skip
    merged.sort(key=lambda c: (-c["votes"], -c["probability"]))

    return merged[:keep]


# ---- the sun -----------------------------------------------------------------------


def ring_zone(
    centre: tuple[float, float], angular_radius_deg: float, width_deg: float, label: str
) -> dict[str, Any]:
    """Every point a fixed angle from ``centre``: where a body stood at a given height."""
    return {
        "kind": "ring",
        "lat": round(centre[0], 4),
        "lon": round(centre[1], 4),
        "radius_km": round(max(0.0, angular_radius_deg) * _KM_PER_DEG, 1),
        "width_km": round(width_deg * _KM_PER_DEG, 1),
        "label": label,
    }


def band_zone(lat_min: float, lat_max: float, label: str) -> dict[str, Any]:
    return {
        "kind": "band",
        "lat_min": round(max(-90.0, lat_min), 2),
        "lat_max": round(min(90.0, lat_max), 2),
        "label": label,
    }


def in_ring(point: tuple[float, float], zone: dict[str, Any]) -> bool:
    apart = haversine_km(point, (zone["lat"], zone["lon"]))

    return abs(apart - zone["radius_km"]) <= zone["width_km"] / 2


def sun_ring(
    when: datetime, elevation_deg: float, tolerance_deg: float = 8.0
) -> dict[str, Any]:
    """Where the sun stood ``elevation_deg`` high at ``when``: a ring around the sub-solar point."""
    from earwitness.modules.provenance import subsolar_point

    return ring_zone(
        subsolar_point(when),
        90.0 - elevation_deg,
        2 * tolerance_deg,
        f"Sun {elevation_deg:.0f}° high at {when.astimezone(timezone.utc):%H:%M} UTC",
    )


def sun_check(
    when: datetime, lat: float, lon: float, inv: dict[str, Any]
) -> dict[str, Any] | None:
    """Does the sun the frames show fit this place at this time?"""
    from earwitness.modules.provenance import sun_position

    expected, _ = sun_position(when, lat, lon)
    lighting = inv.get("lighting")
    seen = inv.get("sun_elevation_deg")

    if lighting == "night" and expected > 6:
        return {
            "ok": False,
            "text": f"night in the frames, but the sun was {expected:.0f}° up there",
        }

    if lighting == "day" and expected < -4:
        return {
            "ok": False,
            "text": f"daylight in the frames, but the sun was {-expected:.0f}° below the horizon there",
        }

    if isinstance(seen, (int, float)) and inv.get("sun_visible"):
        off = abs(expected - float(seen))

        if off <= 12:
            return {
                "ok": True,
                "text": f"sun {expected:.0f}° high there, frames suggest ~{seen:.0f}°",
            }

        if off > 20:
            return {
                "ok": False,
                "text": f"sun {expected:.0f}° high there, frames suggest ~{seen:.0f}°",
            }

        return {
            "ok": None,
            "text": f"sun {expected:.0f}° high there, frames suggest ~{seen:.0f}° (unclear)",
        }

    if lighting in ("day", "night", "twilight"):
        return {"ok": True, "text": f"{lighting} fits (sun {expected:.0f}° there)"}

    return None


# ---- stars -------------------------------------------------------------------------


def gmst_deg(when: datetime) -> float:
    """Greenwich mean sidereal time, degrees."""
    jd = when.astimezone(timezone.utc).timestamp() / 86400 + 2440587.5

    return (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360


def star_zones(
    ra_deg: float, dec_deg: float, tilt_deg: float | None, when: datetime | None
) -> list[dict[str, Any]]:
    """Where a camera could see a star field centred on (RA, Dec).

    Seen at altitude ``a``, the field centre is ``90 - a`` degrees from the
    zenith, so the observer's latitude is within that of the declination; with
    a time the sub-star point is known too, and the locus is a ring.
    """
    spread = 90.0 - (tilt_deg if tilt_deg is not None else 0.0)
    # The altitude is an estimate (camera tilt from the picture): allow 15 degrees.
    slack = min(90.0, spread + 15) if tilt_deg is not None else 90.0
    zones = [
        band_zone(
            dec_deg - slack,
            dec_deg + slack,
            f"Star field at declination {dec_deg:+.0f}° is visible only here",
        )
    ]

    if when is not None and tilt_deg is not None:
        lon = (ra_deg - gmst_deg(when) + 180) % 360 - 180
        zones.append(
            ring_zone(
                (dec_deg, lon),
                spread,
                30.0,
                f"Star field {tilt_deg:.0f}° up at {when:%H:%M} UTC",
            )
        )

    return zones


_solver = None


def _tetra3():
    """The offline plate solver, loaded once (its star database is 8 MB)."""
    global _solver

    if _solver is None:
        import logging

        from earwitness.vendor import tetra3

        logging.getLogger("tetra3.Tetra3").setLevel(logging.WARNING)
        _solver = tetra3.Tetra3()

    return _solver


def solve_stars(
    rgb: np.ndarray, hfov_deg: float = 69.0, sky_fraction: float = 0.6
) -> dict[str, float] | None:
    """Identify the star field in a night picture, offline (tetra3).

    The solver's database covers 10-30 degree fields; a phone frame is about
    65-75 degrees, so crops about a third of the width are tried across the
    upper part of the frame. Returns the solved crop's centre (RA, Dec in
    degrees) and how far above the frame's centre it lies, or None.
    """
    from PIL import Image

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    side = int(min(w * 0.36, h * sky_fraction))
    top = int(h * sky_fraction) - side

    if side < 200:
        return None

    solver = _tetra3()

    for cx in (w // 2, w // 4, 3 * w // 4):
        x0 = max(0, min(w - side, cx - side // 2))
        crop = gray[max(0, top) : max(0, top) + side, x0 : x0 + side]
        fov = 2 * math.degrees(math.atan(side / 2 / f))
        result = solver.solve_from_image(
            Image.fromarray(crop), fov_estimate=fov, fov_max_error=fov * 0.35
        )

        if result.get("RA") is not None and (result.get("Matches") or 0) >= 6:
            cy = max(0, top) + side / 2

            return {
                "ra": float(result["RA"]),
                "dec": float(result["Dec"]),
                "fov": float(result["FOV"]),
                "matches": int(result["Matches"]),
                "centre_px": (x0 + side / 2, cy),
                # How far above the frame centre the solved field lies, degrees.
                "above_centre_deg": math.degrees(math.atan((h / 2 - cy) / f)),
            }

    return None


# ---- ground-level photos -----------------------------------------------------------

_NOT_A_PHOTO = (
    "map",
    "logo",
    "coat of arms",
    "diagram",
    "plan ",
    "chart",
    "svg",
    "flag",
    "locator",
)


def commons_pick(
    geosearch: dict[str, Any], imageinfo: dict[str, Any], n: int
) -> list[dict[str, Any]]:
    """Nearest photographs from a Commons geosearch plus imageinfo response."""
    hits = geosearch.get("query", {}).get("geosearch", [])
    pages = {
        p.get("title"): p for p in imageinfo.get("query", {}).get("pages", {}).values()
    }
    picked = []

    for hit in sorted(hits, key=lambda h: h.get("dist", 1e9)):
        title = hit.get("title", "")

        if any(word in title.lower() for word in _NOT_A_PHOTO):
            continue

        info = (pages.get(title, {}).get("imageinfo") or [{}])[0]

        if info.get("mime") != "image/jpeg" or not info.get("thumburl"):
            continue

        picked.append(
            {"title": title.removeprefix("File:"), "url": info.get("descriptionurl", ""),
             "thumb": info["thumburl"], "dist_m": round(float(hit.get("dist", 0)))}
        )  # fmt: skip

        if len(picked) >= n:
            break

    return picked


async def commons_photos(
    client: httpx.AsyncClient, lat: float, lon: float, radius_m: int, n: int = 2
) -> list[dict[str, Any]]:
    """Geotagged photographs on Wikimedia Commons near a point (no key needed)."""
    api = "https://commons.wikimedia.org/w/api.php"

    try:
        search = (
            await client.get(api, params={
                "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
                "gsradius": max(100, min(10000, radius_m)), "gsnamespace": 6, "gslimit": 25, "format": "json",
            })
        ).json()  # fmt: skip
        titles = [h["title"] for h in search.get("query", {}).get("geosearch", [])][:25]

        if not titles:
            return []

        info = (
            await client.get(api, params={
                "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
                "iiprop": "url|mime", "iiurlwidth": 768, "format": "json",
            })
        ).json()  # fmt: skip
    except httpx.HTTPError, ValueError, KeyError:
        return []

    return commons_pick(search, info, n)


def mapillary_pick(
    response: dict[str, Any], lat: float, lon: float, n: int
) -> list[dict[str, Any]]:
    images = []

    for item in response.get("data", []):
        coords = (item.get("computed_geometry") or item.get("geometry") or {}).get(
            "coordinates"
        )

        if not coords or not item.get("thumb_1024_url"):
            continue

        images.append(
            {"id": item["id"], "thumb": item["thumb_1024_url"],
             "dist_m": round(haversine_km((lat, lon), (coords[1], coords[0])) * 1000),
             "url": f"https://www.mapillary.com/app/?pKey={item['id']}"}
        )  # fmt: skip

    images.sort(key=lambda i: i["dist_m"])
    # Spread out: skip images within 150 m of one already chosen.
    chosen: list[dict[str, Any]] = []

    for image in images:
        if all(abs(image["dist_m"] - c["dist_m"]) > 150 for c in chosen):
            chosen.append(image)

        if len(chosen) >= n:
            break

    return chosen


async def mapillary_photos(
    client: httpx.AsyncClient, lat: float, lon: float, n: int = 2
) -> list[dict[str, Any]]:
    """Street-level images near a point (needs MAPILLARY_TOKEN)."""
    token = os.environ.get("MAPILLARY_TOKEN")

    if not token:
        return []

    d = 0.01  # Mapillary caps the box at 0.01 square degrees

    try:
        response = await client.get(
            "https://graph.mapillary.com/images",
            params={
                "access_token": token, "fields": "id,thumb_1024_url,computed_geometry,geometry",
                "bbox": f"{lon - d / 2},{lat - d / 2},{lon + d / 2},{lat + d / 2}", "limit": 30,
            },
        )  # fmt: skip
        response.raise_for_status()

        return mapillary_pick(response.json(), lat, lon, n)
    except httpx.HTTPError, ValueError, KeyError:
        return []


async def download(client: httpx.AsyncClient, url: str, path: str) -> bool:
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return False

    if response.status_code != 200 or len(response.content) > 6_000_000:
        return False

    with open(path, "wb") as handle:
        handle.write(response.content)

    return cv2.imread(path) is not None


# ---- OpenStreetMap -----------------------------------------------------------------


def overpass_query(lat: float, lon: float, features: list[str]) -> str:
    parts = [
        f"{selector}(around:{FEATURES[f][1]},{lat:.5f},{lon:.5f});"
        for f in features
        if f in FEATURES
        for selector in FEATURES[f][0]
    ]

    return "[out:json][timeout:25];(" + "".join(parts) + ");out tags 500;"


def overpass_found(response: dict[str, Any], features: list[str]) -> dict[str, bool]:
    """Which of ``features`` appear among the elements Overpass returned."""
    found = dict.fromkeys(features, False)

    for element in response.get("elements", []):
        tags = element.get("tags", {})
        kind = element.get("type")

        for feature in features:
            for selector in FEATURES.get(feature, ([], 0))[0]:
                want_type = selector.split("[", 1)[0]
                pairs = [p.strip('"]').split('"="') for p in selector.split("[")[1:]]

                if want_type == kind and all(
                    tags.get(k.strip('"')) == v for k, v in pairs
                ):
                    found[feature] = True

    return found


async def osm_features(
    client: httpx.AsyncClient, lat: float, lon: float, features: list[str]
) -> dict[str, bool] | None:
    """Presence of each feature near the point, from OpenStreetMap (no key needed)."""
    features = [f for f in features if f in FEATURES]

    if not features:
        return None

    query = overpass_query(lat, lon, features)

    for server in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ):
        try:
            response = await client.post(server, data={"data": query}, timeout=40)

            if response.status_code == 200:
                return overpass_found(response.json(), features)
        except httpx.HTTPError, ValueError:
            continue

    return None


def osm_checks(found: dict[str, bool]) -> list[dict[str, Any]]:
    return [
        {
            "ok": present,
            "text": f"{_FEATURE_WORDS[f]} {'mapped nearby' if present else 'seen, but none mapped within ' + format(FEATURES[f][1] / 1000, 'g') + ' km'}",
        }
        for f, present in found.items()
    ]


# ---- terrain skyline ---------------------------------------------------------------

_TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_DEM_ZOOM = 10


def frame_skyline(rgb: np.ndarray, width: int = 320) -> np.ndarray | None:
    """Row of the skyline in each column (fraction of the height), or None.

    Sky is bright, smooth and connected to the top edge; the skyline is where
    it ends. Columns whose top is not sky are NaN. Needs sky across most of
    the frame, or there is no skyline to compare.
    """
    scale = width / rgb.shape[1]
    small = cv2.resize(
        rgb, (width, max(8, round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    h = small.shape[0]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(
        np.float32
    )
    grad = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    )
    top = gray[: max(2, h // 10)]
    bright = gray >= max(60.0, float(np.percentile(top, 20)) - 35)
    smooth = grad < max(18.0, float(np.percentile(grad, 60)))
    mask = (bright & smooth).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(mask)
    top_labels = {int(v) for v in labels[0] if v}

    if not top_labels:
        return None

    sky = np.isin(labels, list(top_labels))
    rows = np.full(width, np.nan)

    for x in range(width):
        column = sky[:, x]

        if not column[0]:
            continue

        ends = np.flatnonzero(~column)
        rows[x] = (ends[0] if ends.size else h) / h

    valid = np.isfinite(rows)

    if valid.mean() < 0.7 or np.nanmedian(rows) > 0.92 or np.nanmedian(rows) < 0.05:
        return None

    return rows


def skyline_angles(
    rows: np.ndarray, aspect: float, hfov_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth offset, elevation angle) in degrees for each skyline column."""
    width = rows.size
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    x = np.arange(width) - width / 2 + 0.5
    y = (0.5 - rows) * width / aspect

    return np.degrees(np.arctan(x / f)), np.degrees(np.arctan(y / f))


def lonlat_to_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 256 * 2**zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

    return x, y


def decode_terrarium(png: np.ndarray) -> np.ndarray:
    """Terrarium tile (BGR as cv2 reads it) to metres."""
    b, g, r = (png[..., i].astype(np.float64) for i in range(3))

    return r * 256 + g + b / 256 - 32768


async def dem_around(
    client: httpx.AsyncClient, lat: float, lon: float, tiles: int = 3
) -> tuple[np.ndarray, float, float, float] | None:
    """Elevation mosaic around a point: (metres, x, y of the point, metres per pixel)."""
    px, py = lonlat_to_pixel(lat, lon, _DEM_ZOOM)
    tx, ty = int(px // 256), int(py // 256)
    half = tiles // 2
    mosaic = np.zeros((256 * tiles, 256 * tiles))

    async def tile(dx: int, dy: int) -> bool:
        url = _TERRARIUM.format(z=_DEM_ZOOM, x=(tx + dx) % (2**_DEM_ZOOM), y=ty + dy)

        try:
            response = await client.get(url)
        except httpx.HTTPError:
            return False

        image = cv2.imdecode(
            np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
        )

        if response.status_code != 200 or image is None:
            return False

        mosaic[
            (dy + half) * 256 : (dy + half + 1) * 256,
            (dx + half) * 256 : (dx + half + 1) * 256,
        ] = decode_terrarium(image)

        return True

    results = await asyncio.gather(
        *(
            tile(dx, dy)
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
        )
    )

    if not all(results):
        return None

    metres = 156543.03392 * math.cos(math.radians(lat)) / 2**_DEM_ZOOM

    return mosaic, px - (tx - half) * 256, py - (ty - half) * 256, metres


def horizon_profile(
    dem: np.ndarray,
    x: float,
    y: float,
    metres_per_px: float,
    eye_m: float = 1.7,
    reach_m: float = 20000,
) -> np.ndarray:
    """Elevation angle of the skyline in each whole-degree azimuth (0 = north, clockwise)."""
    azimuth = np.radians(np.arange(360))[:, None]
    reach = min(reach_m, metres_per_px * (min(dem.shape) / 2 - 2))
    distance = np.geomspace(60.0, reach, 220)[None, :]
    sx = np.clip(
        np.round(x + np.sin(azimuth) * distance / metres_per_px), 0, dem.shape[1] - 1
    ).astype(int)
    sy = np.clip(
        np.round(y - np.cos(azimuth) * distance / metres_per_px), 0, dem.shape[0] - 1
    ).astype(int)
    ground = dem[int(round(y)), int(round(x))] + eye_m
    # Earth curvature less refraction (k = 0.13) lowers distant ground.
    drop = distance**2 / (2 * EARTH_KM * 1000) * 0.87
    angles = np.degrees(np.arctan2(dem[sy, sx] - ground - drop, distance))

    return angles.max(axis=1)


def match_skyline(
    rows: np.ndarray,
    aspect: float,
    profile: np.ndarray,
    fovs: tuple[float, ...] = (55.0, 69.0, 80.0),
) -> dict[str, Any] | None:
    """Best correlation of the frame's skyline with the rendered horizon over headings and FOVs.

    Camera pitch is unknown, so both curves are compared with their means
    removed: the shape of the skyline, not its height in the frame, is matched.
    """
    best: dict[str, Any] | None = None
    valid = np.isfinite(rows)

    for fov in fovs:
        offsets, seen = skyline_angles(rows, aspect, fov)
        offsets, seen = offsets[valid], seen[valid]
        seen_c = seen - seen.mean()

        if seen_c.std() < 0.25:
            return {
                "score": None,
                "relief_seen": float(np.ptp(seen)),
                "relief_there": float(np.ptp(profile)),
            }

        for heading in range(360):
            there = np.interp(
                (heading + offsets) % 360,
                np.arange(361),
                np.append(profile, profile[0]),
            )
            there_c = there - there.mean()

            if there_c.std() < 1e-6:
                continue

            r = float(
                np.dot(seen_c, there_c)
                / (np.linalg.norm(seen_c) * np.linalg.norm(there_c))
            )
            ratio = there_c.std() / seen_c.std()
            score = r if 0.4 <= ratio <= 2.5 else r * 0.5

            if best is None or score > best["score"]:
                best = {
                    "score": round(score, 3), "heading": heading, "fov": fov,
                    "relief_seen": round(float(np.ptp(seen)), 2), "relief_there": round(float(np.ptp(there)), 2),
                }  # fmt: skip

    return best


async def skyline_check(
    client: httpx.AsyncClient, rows: np.ndarray, aspect: float, lat: float, lon: float
) -> dict[str, Any] | None:
    """Compare the frame's skyline with the terrain at and around the candidate."""
    dem = await dem_around(client, lat, lon)

    if dem is None:
        return None

    mosaic, x, y, metres = dem
    best = None
    step = 1500 / metres  # also try 1.5 km around: the candidate is only approximate

    for dx, dy in ((0, 0), (step, 0), (-step, 0), (0, step), (0, -step)):
        profile = horizon_profile(mosaic, x + dx, y + dy, metres)
        result = await asyncio.to_thread(match_skyline, rows, aspect, profile)

        if (
            result
            and result.get("score") is not None
            and (best is None or result["score"] > best["score"])
        ):
            best = result | {"profile": [round(float(v), 2) for v in profile]}

    if best is None:
        return None

    if best["relief_there"] < 0.4 and best["relief_seen"] >= 1.5:
        return best | {
            "ok": False,
            "text": f"hills on the skyline, but the land there is flat ({best['relief_there']:.1f}° of relief)",
        }

    if best["score"] >= 0.85:
        return best | {
            "ok": True,
            "text": f"skyline matches the terrain ({best['score'] * 100:.0f}%, looking {_compass(best['heading'])})",
        }

    return best | {
        "ok": None,
        "text": f"skyline match {best['score'] * 100:.0f}% (inconclusive: trees and buildings hide the terrain)",
    }


def _compass(heading: float) -> str:
    names = [
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    ]

    return names[round(heading / 45) % 8]


# ---- weather -----------------------------------------------------------------------


async def weather_check(
    lat: float, lon: float, when: datetime, seen: str
) -> dict[str, Any] | None:
    from earwitness.modules import weather

    wet_seen, dry_seen = weather._seen_conditions(seen)

    if not (wet_seen or dry_seen):
        return None

    record = await weather._hourly(lat, lon, when)

    if record is None:
        return None

    sky = weather._CODES.get(int(record.get("weather_code") or 0), "unknown")
    rain = float(record.get("precipitation") or 0)
    wet = rain >= 0.5 or sky.endswith(("rain", "drizzle", "showers"))
    dry = rain == 0 and int(record.get("weather_code") or 0) <= 2

    if (wet and dry_seen) or (dry and wet_seen):
        return {
            "ok": False,
            "text": f"weather there was {sky}, the frames look {'dry' if dry_seen else 'wet'}",
        }

    return {"ok": True, "text": f"weather there was {sky}, which fits"}
