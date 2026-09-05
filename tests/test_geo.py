from shipwreck_sat_finder.geo import SceneBounds


def test_roundtrip():
    b = SceneBounds(36.88, 36.96, -76.02, -75.94, 100, 100)
    lat, lon = b.pixel_to_latlon(50, 50)
    x, y = b.latlon_to_pixel(lat, lon)
    assert abs(x - 50) < 1e-6
    assert abs(y - 50) < 1e-6
