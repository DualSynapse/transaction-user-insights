"""geopandas: province, regency/city, region enrichment via point-in-polygon."""
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def parse_coordinates(geo_series: pd.Series) -> pd.DataFrame:
    """Parse 'lat, lon' strings into separate float columns."""
    parts = geo_series.str.split(",", expand=True)
    lat = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    lon = pd.to_numeric(parts[1].str.strip(), errors="coerce") if parts.shape[1] > 1 else pd.Series(np.nan, index=geo_series.index)
    return pd.DataFrame({"lat": lat, "lon": lon})


def _ensure_boundary_file(geo_cfg: dict, external_dir: Path) -> Path | None:
    boundary_path = external_dir / geo_cfg["boundary_file"]
    if boundary_path.exists():
        return boundary_path
    external_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading administrative boundary file from %s", geo_cfg["boundary_url"])
        req = urllib.request.Request(geo_cfg["boundary_url"], headers={"User-Agent": "transaction-user-insights"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        boundary_path.write_bytes(data)
        return boundary_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download boundary file (%s); will use centroid fallback.", exc)
        return None


def _normalize_names(gdf, geo_cfg: dict):
    province_col = geo_cfg["province_col"]
    regency_col = geo_cfg["regency_col"]
    type_col = geo_cfg["type_col"]

    norm = geo_cfg["name_normalization"]
    province_map = norm.get("provinces", {})
    type_map = norm.get("city_type_map", {})

    gdf = gdf.copy()
    gdf["province"] = gdf[province_col].replace(province_map)
    gdf["city_type"] = gdf[type_col].replace(type_map)

    prefix = gdf["city_type"].map({"Kota": "Kota", "Kabupaten": "Kabupaten"}).fillna("Kabupaten")
    already_prefixed = gdf[regency_col].str.startswith(("Kota ", "Kabupaten "))
    gdf["city"] = np.where(already_prefixed, gdf[regency_col], prefix + " " + gdf[regency_col])

    return gdf


def _region_group(city: str, region_groups: dict) -> str | None:
    for region, cities in region_groups.items():
        if city in cities:
            return region
    return None


def _centroid_fallback(coords: pd.DataFrame, geo_cfg: dict) -> pd.DataFrame:
    """Match every point to the nearest fallback centroid (used when no boundary file at all)."""
    centroids = geo_cfg["fallback_centroids"]
    names = list(centroids.keys())
    clat = np.array([centroids[n]["lat"] for n in names])
    clon = np.array([centroids[n]["lon"] for n in names])

    result = pd.DataFrame(index=coords.index, columns=["province", "city", "city_type", "geo_match_method"])
    valid = coords["lat"].notna() & coords["lon"].notna()

    lat = coords.loc[valid, "lat"].to_numpy()
    lon = coords.loc[valid, "lon"].to_numpy()
    # squared-distance in degrees is sufficient for picking the nearest of a handful of centroids
    d2 = (lat[:, None] - clat[None, :]) ** 2 + (lon[:, None] - clon[None, :]) ** 2
    nearest_idx = d2.argmin(axis=1)

    result.loc[valid, "city"] = [names[i] for i in nearest_idx]
    result.loc[valid, "province"] = [centroids[names[i]]["province"] for i in nearest_idx]
    result.loc[valid, "city_type"] = [centroids[names[i]]["city_type"] for i in nearest_idx]
    result.loc[valid, "geo_match_method"] = "centroid_fallback"

    result.loc[~valid, "geo_match_method"] = "unmatched"
    return result


def enrich_geo(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    coords = parse_coordinates(df["geo_location"])
    df["lat"] = coords["lat"]
    df["lon"] = coords["lon"]
    df["dq_geo_unparseable"] = df["lat"].isna() | df["lon"].isna()
    df["dq_geo_outside_indonesia"] = (
        df["lat"].notna() & df["lon"].notna()
        & ~df["lat"].between(-11, 6) & ~df["lon"].between(95, 141)
    )

    geo_cfg = cfg["geo"]
    external_dir = cfg.path("external_dir")

    boundary_path = None
    if GEOPANDAS_AVAILABLE:
        boundary_path = _ensure_boundary_file(geo_cfg, external_dir)

    unique_coords = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

    if GEOPANDAS_AVAILABLE and boundary_path is not None:
        matched = _match_with_boundary(unique_coords, boundary_path, geo_cfg)
    else:
        matched = _centroid_fallback(unique_coords, geo_cfg)
        matched = pd.concat([unique_coords, matched], axis=1)

    matched["region_group"] = matched["city"].apply(
        lambda c: _region_group(c, geo_cfg["region_groups"]) if pd.notna(c) else None
    )

    df = df.merge(matched, on=["lat", "lon"], how="left")
    df["geo_match_method"] = df["geo_match_method"].fillna("unmatched")
    df["dq_geo_unmatched"] = df["geo_match_method"] == "unmatched"

    return df


def _match_with_boundary(unique_coords: pd.DataFrame, boundary_path: Path, geo_cfg: dict) -> pd.DataFrame:
    gdf_boundary = gpd.read_file(boundary_path)
    gdf_boundary = _normalize_names(gdf_boundary, geo_cfg)
    gdf_boundary = gdf_boundary.set_crs(epsg=4326, allow_override=True)

    valid = unique_coords["lat"].notna() & unique_coords["lon"].notna()
    points = unique_coords.loc[valid].copy()
    geometry = [Point(lon, lat) for lat, lon in zip(points["lat"], points["lon"])]
    gdf_points = gpd.GeoDataFrame(points, geometry=geometry, crs="EPSG:4326")

    keep_cols = ["province", "city", "city_type"]

    # 1) within
    joined = gpd.sjoin(gdf_points, gdf_boundary[keep_cols + ["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined["geo_match_method"] = np.where(joined["province"].notna(), "within", None)

    # 2) nearest, for points that didn't match "within"
    unmatched_mask = joined["province"].isna()
    if unmatched_mask.any():
        unmatched_points = gdf_points.loc[unmatched_mask]
        # project to a metric CRS for distance-based nearest join
        proj_points = unmatched_points.to_crs(epsg=3857)
        proj_boundary = gdf_boundary[keep_cols + ["geometry"]].to_crs(epsg=3857)
        nearest = gpd.sjoin_nearest(
            proj_points, proj_boundary, how="left", distance_col="dist_m",
            max_distance=geo_cfg["nearest_max_km"] * 1000,
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        for col in keep_cols:
            joined.loc[nearest.index, col] = nearest[col]
        matched_nearest = nearest["province"].notna()
        joined.loc[nearest.index[matched_nearest], "geo_match_method"] = "nearest"

    joined["geo_match_method"] = joined["geo_match_method"].fillna("unmatched")

    result = unique_coords.copy()
    for col in keep_cols + ["geo_match_method"]:
        result[col] = None
    result.loc[valid, keep_cols + ["geo_match_method"]] = joined[keep_cols + ["geo_match_method"]].values
    result.loc[~valid, "geo_match_method"] = "unmatched"

    return result
