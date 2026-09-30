import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.cleaning.geo_enrichment import _centroid_fallback, _match_with_boundary, _normalize_names, parse_coordinates


def _make_boundary_file(tmp_path):
    # A 1x1 degree square "city" centered at (0, 0) named after normalization rules.
    square = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    gdf = gpd.GeoDataFrame(
        {
            "NAME_1": ["Jakarta Raya"],
            "NAME_2": ["Jakarta Pusat"],
            "TYPE_2": ["Kotamadya"],
        },
        geometry=[square],
        crs="EPSG:4326",
    )
    path = tmp_path / "boundary.json"
    gdf.to_file(path, driver="GeoJSON")
    return path


GEO_CFG = {
    "province_col": "NAME_1",
    "regency_col": "NAME_2",
    "type_col": "TYPE_2",
    "nearest_max_km": 200,
    "name_normalization": {
        "provinces": {"Jakarta Raya": "DKI Jakarta"},
        "city_type_map": {"Kotamadya": "Kota", "Kabupaten": "Kabupaten"},
    },
    "fallback_centroids": {
        "Kota Jakarta Pusat": {"province": "DKI Jakarta", "lat": 0.0, "lon": 0.0, "city_type": "Kota"},
        "Kota Faraway": {"province": "Far Province", "lat": 10.0, "lon": 10.0, "city_type": "Kota"},
    },
}


def test_parse_coordinates():
    coords = parse_coordinates(pd.Series(["-6.2, 106.8", "1.0,2.0"]))
    assert coords["lat"].tolist() == pytest.approx([-6.2, 1.0])
    assert coords["lon"].tolist() == pytest.approx([106.8, 2.0])


def test_name_normalization():
    gdf = gpd.GeoDataFrame(
        {"NAME_1": ["Jakarta Raya"], "NAME_2": ["Jakarta Pusat"], "TYPE_2": ["Kotamadya"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])],
        crs="EPSG:4326",
    )
    normed = _normalize_names(gdf, GEO_CFG)
    assert normed["province"].iloc[0] == "DKI Jakarta"
    assert normed["city"].iloc[0] == "Kota Jakarta Pusat"


def test_within_and_nearest_match(tmp_path):
    boundary_path = _make_boundary_file(tmp_path)
    # one point inside the square, one just outside it (within nearest_max_km)
    coords = pd.DataFrame({"lat": [0.0, 0.6], "lon": [0.0, 0.0]})

    matched = _match_with_boundary(coords, boundary_path, GEO_CFG)

    inside = matched.iloc[0]
    assert inside["geo_match_method"] == "within"
    assert inside["city"] == "Kota Jakarta Pusat"

    outside = matched.iloc[1]
    assert outside["geo_match_method"] == "nearest"
    assert outside["city"] == "Kota Jakarta Pusat"


def test_centroid_fallback():
    coords = pd.DataFrame({"lat": [0.01, 10.01, None], "lon": [0.01, 10.01, None]})
    result = _centroid_fallback(coords, GEO_CFG)

    assert result.loc[0, "geo_match_method"] == "centroid_fallback"
    assert result.loc[0, "city"] == "Kota Jakarta Pusat"
    assert result.loc[1, "city"] == "Kota Faraway"
    assert result.loc[2, "geo_match_method"] == "unmatched"
