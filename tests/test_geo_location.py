import pytest

from earwitness.modules.geo import _has_location


@pytest.mark.parametrize(
    ("lat", "lon", "radius", "expected"),
    [
        (0, 0, 20000, False),
        (0, 0, 50, True),
        (51.5, -0.1, 50, True),
        (51.5, -0.1, 2500, False),
        (91, 0, 50, False),
        (0, float("nan"), 50, False),
        (None, 0, 50, False),
        (0, 0, -1, False),
    ],
)
def test_location_requires_a_valid_specific_estimate(lat, lon, radius, expected):
    assert _has_location(lat, lon, radius) is expected
