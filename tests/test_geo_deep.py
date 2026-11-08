"""Deep geolocation helpers: the maths and the parsing, without the network."""

from datetime import datetime
from datetime import timezone

import numpy as np

from earwitness.modules import geo_deep as g
from earwitness.modules.provenance import subsolar_point
from earwitness.modules.provenance import sun_position

EDINBURGH = (55.95, -3.19)


def test_subsolar_point_has_the_sun_overhead():
    for when in (
        datetime(2026, 6, 21, 12, tzinfo=timezone.utc),
        datetime(2026, 11, 8, 15, 30, tzinfo=timezone.utc),
    ):
        lat, lon = subsolar_point(when)

        assert sun_position(when, lat, lon)[0] > 89.5


def test_sun_ring_passes_through_the_true_place():
    when = datetime(2026, 8, 1, 13, tzinfo=timezone.utc)
    elevation, _ = sun_position(when, *EDINBURGH)
    ring = g.sun_ring(when, elevation)

    assert g.in_ring(EDINBURGH, ring)
    # Sydney saw no such sun at that moment.
    assert not g.in_ring((-33.87, 151.21), ring)


def test_sun_check_catches_night_against_a_high_sun():
    when = datetime(2026, 8, 1, 13, tzinfo=timezone.utc)

    assert g.sun_check(when, *EDINBURGH, {"lighting": "night"})["ok"] is False
    assert (
        g.sun_check(
            when,
            *EDINBURGH,
            {"lighting": "day", "sun_visible": True, "sun_elevation_deg": 48},
        )["ok"]
        is True
    )


def test_star_band_bounds_the_latitude():
    # A field near the south celestial pole cannot be seen from Edinburgh.
    band = g.star_zones(200.0, -60.0, None, None)[0]

    assert band["lat_max"] == 30.0 and band["lat_min"] == -90.0


def test_cluster_counts_agreeing_passes():
    a = {"place": "Peebles", "latitude": 55.65, "longitude": -3.19, "probability": 0.5}
    b = {
        "place": "near Peebles",
        "latitude": 55.70,
        "longitude": -3.25,
        "probability": 0.3,
    }
    c = {"place": "Keswick", "latitude": 54.60, "longitude": -3.13, "probability": 0.4}

    merged = g.cluster([[a, c], [b], [dict(a, probability=0.2)]])

    assert merged[0]["place"] == "Peebles"
    assert merged[0]["votes"] == 3 and merged[0]["passes"] == 3
    assert merged[0]["probability"] == round((0.5 + 0.3 + 0.2) / 3, 3)
    assert merged[1]["place"] == "Keswick" and merged[1]["votes"] == 1


def test_overpass_found_matches_tags():
    response = {
        "elements": [
            {
                "type": "node",
                "tags": {"power": "generator", "generator:source": "wind"},
            },
            {"type": "way", "tags": {"natural": "coastline"}},
            {"type": "node", "tags": {"natural": "coastline"}},  # wrong element type
        ]
    }

    found = g.overpass_found(response, ["wind_turbine", "coastline", "railway"])

    assert found == {"wind_turbine": True, "coastline": True, "railway": False}
    assert "around:12000" in g.overpass_query(55.0, -3.0, ["wind_turbine"])
    assert [c["ok"] for c in g.osm_checks(found)] == [True, True, False]


def test_commons_pick_skips_maps_and_non_photos():
    search = {"query": {"geosearch": [
        {"title": "File:Location map of Peebles.jpg", "dist": 10},
        {"title": "File:Track over the moor.jpg", "dist": 300},
        {"title": "File:Drawing.svg", "dist": 50},
    ]}}  # fmt: skip
    info = {"query": {"pages": {
        "1": {"title": "File:Location map of Peebles.jpg", "imageinfo": [{"mime": "image/jpeg", "thumburl": "u1"}]},
        "2": {"title": "File:Track over the moor.jpg", "imageinfo": [{"mime": "image/jpeg", "thumburl": "u2", "descriptionurl": "d2"}]},
        "3": {"title": "File:Drawing.svg", "imageinfo": [{"mime": "image/svg+xml", "thumburl": "u3"}]},
    }}}  # fmt: skip

    assert g.commons_pick(search, info, 2) == [
        {"title": "Track over the moor.jpg", "url": "d2", "thumb": "u2", "dist_m": 300}
    ]


def test_mapillary_pick_spreads_out():
    response = {"data": [
        {"id": "1", "thumb_1024_url": "a", "computed_geometry": {"coordinates": [-3.19, 55.9501]}},
        {"id": "2", "thumb_1024_url": "b", "computed_geometry": {"coordinates": [-3.19, 55.9502]}},
        {"id": "3", "thumb_1024_url": "c", "computed_geometry": {"coordinates": [-3.19, 55.9540]}},
    ]}  # fmt: skip

    assert [i["id"] for i in g.mapillary_pick(response, *EDINBURGH, 2)] == ["1", "3"]


def _ridge_dem() -> np.ndarray:
    """Flat ground with a 600 m ridge 5 km to the north."""
    dem = np.zeros((400, 400))
    dem[130:140, :] = 600.0  # 100 m per pixel: rows 130-140 are ~6 km north of row 200

    return dem


def test_horizon_profile_sees_the_ridge_to_the_north():
    profile = g.horizon_profile(_ridge_dem(), 200, 200, 100.0)

    assert profile[0] > 4  # atan(600 / ~6500 m) is about 5 degrees
    assert abs(profile[180]) < 0.5
    assert profile.argmax() in (0, 359, 1)


def test_skyline_from_frame_and_match():
    profile = (
        g.horizon_profile(_ridge_dem(), 200, 200, 100.0)
        + np.sin(np.radians(np.arange(360)) * 6) * 2
    )
    w, h = 400, 300
    offsets, _ = g.skyline_angles(np.zeros(w), w / h, 69.0)
    there = np.interp(
        (30 + offsets) % 360, np.arange(361), np.append(profile, profile[0])
    )
    f = (w / 2) / np.tan(np.radians(69.0) / 2)
    rows = 0.5 - np.tan(np.radians(there - there.mean())) * f / h
    image = np.full((h, w, 3), (80, 100, 50), np.uint8)

    for x in range(w):
        image[: int(rows[x] * h), x] = (175, 205, 240)

    seen = g.frame_skyline(image, width=w)

    assert seen is not None and np.isfinite(seen).mean() > 0.95

    best = g.match_skyline(seen, w / h, profile)

    assert best["score"] > 0.95
    assert abs(best["heading"] - 30) <= 2


def test_frame_without_sky_has_no_skyline():
    rng = np.random.default_rng(1)

    assert g.frame_skyline(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)) is None


def test_inventory_text_drops_empty_fields():
    text = g.inventory_text({"road": "single track, no lines", "vehicles": "none", "hemisphere_hint": "unknown",
                             "strongest_clues": ["dry-stone walls"]})  # fmt: skip

    assert "Road: single track" in text
    assert "Vehicles" not in text and "Hemisphere" not in text
    assert "dry-stone walls" in text


def _night_photo(
    ra_deg: float, dec_deg: float, w: int = 1800, h: int = 1200, hfov: float = 69.0
):
    """A synthetic night photo: real stars (from the solver's own catalogue) around (RA, Dec)."""
    stars = g._tetra3().star_table
    ra0, dec0 = np.radians(ra_deg), np.radians(dec_deg)
    centre = np.array(
        [np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)]
    )
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.cross(centre, east)
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    v = stars[:, 2:5]
    dot = v @ centre
    keep = dot > 0.3
    px = w / 2 + f * (v[keep] @ east) / dot[keep]
    py = h / 2 - f * (v[keep] @ north) / dot[keep]
    image = np.random.default_rng(0).normal(14, 3, (h, w))
    yy, xx = np.mgrid[0:13, 0:13]

    for x, y, mag in zip(px, py, stars[keep, 5]):
        if 6 <= x < w - 7 and 6 <= y < h - 7 and mag < 6.5:
            x0, y0 = int(x) - 6, int(y) - 6
            image[y0 : y0 + 13, x0 : x0 + 13] += min(
                240.0, 255 * 10 ** (-0.4 * (mag - 1))
            ) * np.exp(-((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2) / 3.0)

    gray = np.clip(image, 0, 255).astype(np.uint8)

    def direction(x: float, y: float) -> tuple[float, float]:
        d = centre * f + east * (x - w / 2) - north * (y - h / 2)
        d /= np.linalg.norm(d)
        return np.degrees(np.arctan2(d[1], d[0])) % 360, np.degrees(np.arcsin(d[2]))

    return np.dstack([gray] * 3), direction


def test_solve_stars_identifies_orion_offline():
    photo, direction = _night_photo(83.8, -5.4)

    solved = g.solve_stars(photo)

    assert solved is not None and solved["matches"] >= 6
    ra, dec = direction(*solved["centre_px"])
    assert abs(solved["ra"] - ra) < 1.0 and abs(solved["dec"] - dec) < 1.0


def test_solve_stars_gives_up_on_a_starless_sky():
    assert g.solve_stars(np.full((1200, 1800, 3), 20, np.uint8)) is None


def _observer_photo(lat, lst, pitch=20.0, azimuth=0.0, w=1800, h=1200, hfov=69.0):
    """The sky a camera at latitude ``lat`` sees at local sidereal time ``lst``.

    Stars are placed by their real altitude and azimuth, about as bright as a
    phone's night mode records them; below the level horizon is dark,
    starless ground. Returns the picture and the horizon row
    (fraction of the height).
    """
    stars = g._tetra3().star_table
    phi, theta = np.radians(lat), np.radians(lst)
    ra, dec = stars[:, 0], stars[:, 1]
    hour = theta - ra
    alt = np.arcsin(
        np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour)
    )
    az = np.arctan2(
        -np.cos(dec) * np.sin(hour),
        np.sin(dec) * np.cos(phi) - np.cos(dec) * np.cos(hour) * np.sin(phi),
    )
    # East, north, up.
    v = np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], 1)
    a, p = np.radians(azimuth), np.radians(pitch)
    forward = np.array([np.sin(a) * np.cos(p), np.cos(a) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    up = np.cross(right, forward)
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    depth = v @ forward
    seen = (alt > 0) & (depth > 0.2)
    px = w / 2 + f * (v[seen] @ right) / depth[seen]
    py = h / 2 - f * (v[seen] @ up) / depth[seen]
    horizon = h / 2 + f * np.tan(p)
    image = np.random.default_rng(1).normal(16, 3, (h, w))
    yy, xx = np.mgrid[0:13, 0:13]

    for x, y, mag in zip(px, py, stars[seen, 5]):
        if 6 <= x < w - 7 and 6 <= y < min(h, horizon) - 7 and mag < 6.5:
            x0, y0 = int(x) - 6, int(y) - 6
            image[y0 : y0 + 13, x0 : x0 + 13] += min(
                240.0, 255 * 10 ** (-0.4 * (mag - 3))
            ) * np.exp(-((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2) / 3.0)

    image[int(horizon) :] = np.random.default_rng(2).normal(
        5, 1.5, (h - int(horizon), w)
    )
    gray = np.clip(image, 0, 255).astype(np.uint8)

    return np.dstack([gray] * 3), horizon / h


def test_pole_height_gives_the_latitude_without_a_date():
    photo, horizon = _observer_photo(lat=47.0, lst=300.0)
    solved = g.solve_stars(photo)

    assert solved is not None

    # The analyst's guess of the horizon is a little off; the brightness step fixes it.
    guess = horizon - 0.03
    line = g.refine_horizon(photo[..., 0], guess, guess)
    assert abs(line[0] - horizon) < 0.01 and abs(line[1] - horizon) < 0.01

    fix = g.zenith_from_stars(solved, photo.shape[1], photo.shape[0], line)

    assert abs(fix["latitude"] - 47.0) < 2.0
    assert abs(((fix["sidereal_deg"] - 300.0 + 180) % 360) - 180) < 3.0


def test_star_fix_with_a_time_gives_a_spot():
    when = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
    lon = -110.0  # Montana
    lst = (g.gmst_deg(when) + lon) % 360
    photo, horizon = _observer_photo(lat=46.0, lst=lst, pitch=15.0)
    fix = g.zenith_from_stars(
        g.solve_stars(photo), photo.shape[1], photo.shape[0], (horizon, horizon)
    )

    band, spot = g.star_fix_zones(fix, when)

    assert band["lat_min"] < 46.0 < band["lat_max"]
    assert g.in_ring((46.0, lon), spot)
    assert not g.in_ring((36.2, 44.0), spot)  # Erbil
