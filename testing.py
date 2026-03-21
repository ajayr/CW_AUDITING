# testing.py
"""
Pytest file – Input & Inference section
Tests 1–2: time string → minutes conversion using DataLoaderClass._TimeToSeconds
"""

import pytest
import numpy as np

# Change 'dataloader' to match your actual file name
# Examples:
#   from dataloader import DataLoaderClass
#   from DataLoader import DataLoaderClass
from analytics.DataLoader import DataLoaderClass   # ← ← ← ADJUST THIS LINE


# Helper wrapper – mimics how your inference code probably uses this method
def time_str_to_minutes(time_str):
    """
    Calls the internal _TimeToSeconds method and converts result to minutes.
    Returns float minutes or np.nan if invalid/unparseable.
    """
    # We create a dummy instance just to access the method (no real file needed)
    dummy_loader = DataLoaderClass.__new__(DataLoaderClass)
    seconds = dummy_loader._TimeToSeconds(time_str)
    if np.isnan(seconds):
        return np.nan
    return seconds / 60.0


# ───────────────────────────────────────────────
# Test 1: Normal cases – valid HH:MM strings
# ───────────────────────────────────────────────
def test_time_conversion_normal():
    assert time_str_to_minutes("00:08:45") == pytest.approx(8.75)       # 525s / 60
    assert time_str_to_minutes("00:14:30") == pytest.approx(14.5)       # 870s / 60
    assert time_str_to_minutes("00:00:15") == pytest.approx(0.25)       # 15s / 60
    assert time_str_to_minutes("00:23:00") == pytest.approx(23.0)       # 1380s / 60
    assert time_str_to_minutes("00:19:22") == pytest.approx((19*60+22)/60.0)  # 1162s / 60
    assert time_str_to_minutes("00:05:09") == pytest.approx(309.0/60.0)       # 5.15 min


# ───────────────────────────────────────────────
# Test 2: Boundary minimum – 00:00
# ───────────────────────────────────────────────
def test_time_conversion_boundary_min():
    assert time_str_to_minutes("00:00") == 0.0


# Optional: quick smoke test that invalid inputs become nan
# (you can move this to test 4/5 later)
def test_time_conversion_invalid_becomes_nan():
    assert np.isnan(time_str_to_minutes("abc"))
    assert np.isnan(time_str_to_minutes(""))
    assert np.isnan(time_str_to_minutes("--"))


def test_time_conversion_two_part_string():
    # "25:00" is valid MM:SS → 25 minutes
    assert time_str_to_minutes("25:00") == pytest.approx(25.0)


# ───────────────────────────────────────────────
# Tests: DateHierarchyTree matches pandas groupby
# ───────────────────────────────────────────────
import pandas as pd
from pathlib import Path
from analytics.DateHierarchyTree import DateHierarchyTree


def _load_analytics():
    from analytics.RunningAnalytics import RunningAnalyticsClass
    csv_path = Path(__file__).resolve().parent / "data" / "GarminFullRunning.csv"
    return RunningAnalyticsClass(csv_path)


def _pandas_monthly(df):
    return (
        df.groupby("YearMonth")
        .agg(
            total_distance=("Distance",     "sum"),
            avg_hr=        ("Avg HR",       "mean"),
            avg_pace_s=    ("Avg Pace_sec", "mean"),
            run_count=     ("Distance",     "count"),
            total_calories=("Calories",     "sum"),
        )
        .reset_index()
    )


def _pandas_yearly(df):
    return (
        df.groupby("year")
        .agg(
            total_distance=("Distance",     "sum"),
            avg_hr=        ("Avg HR",       "mean"),
            avg_pace_s=    ("Avg Pace_sec", "mean"),
            run_count=     ("Distance",     "count"),
            total_calories=("Calories",     "sum"),
        )
        .reset_index()
    )


def test_tree_monthly_matches_pandas():
    ra = _load_analytics()
    expected = _pandas_monthly(ra.df)
    actual = ra.MonthlySummary()
    pd.testing.assert_frame_equal(actual, expected)


def test_tree_yearly_matches_pandas():
    ra = _load_analytics()
    expected = _pandas_yearly(ra.df)
    actual = ra.YearlySummary()
    pd.testing.assert_frame_equal(actual, expected)


# ───────────────────────────────────────────────
# Tests: OOP – Abstract base classes & polymorphism
# ───────────────────────────────────────────────
from analytics.base_processor import BaseDataProcessor
from analytics.chart_generators import (
    ChartGenerator,
    DistanceOverTimeChart,
    EfficiencyOverTimeChart,
    WeeklyLoadVsPaceChart,
)


def test_dataloader_is_base_data_processor():
    ra = _load_analytics()
    assert isinstance(ra, BaseDataProcessor)


def test_joined_loader_is_base_data_processor():
    from analytics.JoinedDataLoader import JoinedDataLoaderClass
    csv_path = Path(__file__).resolve().parent / "data" / "JoinedRunWeather.csv"
    loader = JoinedDataLoaderClass(csv_path)
    assert isinstance(loader, BaseDataProcessor)


def test_chart_generators_are_polymorphic():
    charts = [DistanceOverTimeChart(), EfficiencyOverTimeChart(), WeeklyLoadVsPaceChart()]
    for chart in charts:
        assert isinstance(chart, ChartGenerator)


def test_abstract_base_processor_not_instantiable():
    with pytest.raises(TypeError):
        BaseDataProcessor()


def test_abstract_chart_generator_not_instantiable():
    with pytest.raises(TypeError):
        ChartGenerator()


# ───────────────────────────────────────────────
# Tests: Custom mergesort
# ───────────────────────────────────────────────
from analytics.mergesort import mergesort, mergesort_dataframe


def test_mergesort_integers():
    assert mergesort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]


def test_mergesort_strings():
    assert mergesort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_mergesort_with_key():
    items = [{"name": "b", "val": 2}, {"name": "a", "val": 1}, {"name": "c", "val": 3}]
    result = mergesort(items, key=lambda x: x["val"])
    assert [r["name"] for r in result] == ["a", "b", "c"]


def test_mergesort_reverse():
    assert mergesort([3, 1, 2], reverse=True) == [3, 2, 1]


def test_mergesort_empty_and_single():
    assert mergesort([]) == []
    assert mergesort([42]) == [42]


def test_mergesort_dataframe_matches_sort_values():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-03-01", "2024-01-15", "2024-02-10"]),
        "value": [30, 10, 20],
    })
    expected = df.sort_values("Date").reset_index(drop=True)
    actual = mergesort_dataframe(df, by="Date")
    pd.testing.assert_frame_equal(actual, expected)


def test_mergesort_dataframe_with_nan():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-03-01", pd.NaT, "2024-01-15"]),
        "value": [30, 20, 10],
    })
    expected = df.sort_values("Date").reset_index(drop=True)
    actual = mergesort_dataframe(df, by="Date")
    pd.testing.assert_frame_equal(actual, expected)


# ───────────────────────────────────────────────
# Tests: Custom HashTable
# ───────────────────────────────────────────────
from analytics.hashtable import HashTable


def test_hashtable_put_and_get():
    ht = HashTable({"a": 1, "b": 2, "c": 3})
    assert ht.get("a") == 1
    assert ht.get("b") == 2
    assert ht.get("c") == 3


def test_hashtable_get_default():
    ht = HashTable({"x": 10})
    assert ht.get("missing") is None
    assert ht.get("missing", 42) == 42


def test_hashtable_contains():
    ht = HashTable({"key": "val"})
    assert "key" in ht
    assert "other" not in ht


def test_hashtable_collision_handling():
    ht = HashTable(capacity=4)
    for i in range(10):
        ht.put(f"key{i}", i)
    for i in range(10):
        assert ht.get(f"key{i}") == i


def test_hashtable_overwrite():
    ht = HashTable()
    ht.put("k", 1)
    ht.put("k", 2)
    assert ht.get("k") == 2
    assert len(ht) == 1


def test_hashtable_matches_dict_for_injury_course_maps():
    injury_dict = {"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3}
    injury_ht = HashTable(injury_dict)
    for key, val in injury_dict.items():
        assert injury_ht.get(key) == val
    assert injury_ht.get("Unknown", 0) == 0

    course_dict = {"Flat": 0, "Mixed": 1, "Hilly": 2}
    course_ht = HashTable(course_dict)
    for key, val in course_dict.items():
        assert course_ht.get(key) == val


# ───────────────────────────────────────────────
# Tests: Deque-based rolling mean
# ───────────────────────────────────────────────
from analytics.chart_generators import _deque_rolling_mean


def test_deque_rolling_mean_matches_pandas():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    expected = pd.Series(values).rolling(window=4, min_periods=1).mean().tolist()
    actual = _deque_rolling_mean(values, window=4)
    np.testing.assert_array_almost_equal(actual, expected)


def test_deque_rolling_mean_single_value():
    assert _deque_rolling_mean([5.0], window=4) == [5.0]


def test_deque_rolling_mean_empty():
    assert _deque_rolling_mean([], window=4) == []


# ═══════════════════════════════════════════════
# DataLoader pipeline tests
# ═══════════════════════════════════════════════

def _make_dummy_loader():
    """Create a DataLoaderClass instance without loading a file."""
    return DataLoaderClass.__new__(DataLoaderClass)


def test_pace_to_seconds():
    loader = _make_dummy_loader()
    assert loader._PaceToSeconds("5:30") == 330
    assert loader._PaceToSeconds("0:00") == 0
    assert np.isnan(loader._PaceToSeconds("--"))
    assert np.isnan(loader._PaceToSeconds("abc"))
    assert np.isnan(loader._PaceToSeconds(""))
    assert np.isnan(loader._PaceToSeconds("n/a"))


def test_convert_dates():
    loader = _make_dummy_loader()
    df = pd.DataFrame({"Date": ["2024-01-15", "invalid", "2024-03-01"]})
    df = loader.ConvertDates(df)
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert pd.notna(df["Date"].iloc[0])
    assert pd.isna(df["Date"].iloc[1])
    assert pd.notna(df["Date"].iloc[2])


def test_convert_numeric_columns():
    loader = _make_dummy_loader()
    df = pd.DataFrame({
        "Distance": ["10.5", "--", "1,234", "n/a", "5.0"],
        "Avg HR": ["150", "nan", "160", "", "155"],
    })
    df = loader.ConvertNumericColumns(df)
    assert df["Distance"].iloc[0] == pytest.approx(10.5)
    assert np.isnan(df["Distance"].iloc[1])
    assert df["Distance"].iloc[2] == pytest.approx(1234.0)
    assert np.isnan(df["Distance"].iloc[3])
    assert df["Avg HR"].iloc[0] == pytest.approx(150.0)
    assert np.isnan(df["Avg HR"].iloc[1])


def test_remove_duplicate_columns():
    loader = _make_dummy_loader()
    df = pd.DataFrame({"Col": [1], "Col.1": [2], "Other": [3]})
    df = loader.RemoveDuplicateColumns(df)
    assert "Col" in df.columns
    assert "Col.1" not in df.columns
    assert "Other" in df.columns


def _make_synthetic_df():
    """Minimal synthetic DataFrame with all columns needed by process()."""
    return pd.DataFrame({
        "Date": ["2024-06-15", "2024-06-20", "2024-07-01"],
        "Distance": ["5.0", "10.0", "8.0"],
        "Calories": ["300", "600", "480"],
        "Avg HR": ["150", "160", "155"],
        "Max HR": ["170", "180", "175"],
        "Avg Pace": ["5:30", "6:00", "5:45"],
        "Time": ["0:27:30", "1:00:00", "0:46:00"],
    })


def test_create_extra_columns():
    loader = _make_dummy_loader()
    df = _make_synthetic_df()
    df = loader.ConvertDates(df)
    df = loader.ConvertNumericColumns(df)
    df = loader.ConvertPaceColumns(df)
    df = loader.ConvertTimeColumns(df)
    df = loader.CreateExtraColumns(df)

    assert "hr_efficiency" in df.columns
    assert "speed_kmh" in df.columns
    assert "year" in df.columns
    assert "YearMonth" in df.columns
    assert "duration_min" in df.columns
    assert df["year"].iloc[0] == 2024
    assert df["YearMonth"].iloc[0] == "2024-06"


def test_process_pipeline_end_to_end():
    df = _make_synthetic_df()
    instance = DataLoaderClass.FromDataframe(df)
    assert "Date" in instance.df.columns
    assert "Avg Pace_sec" in instance.df.columns
    assert "Time_sec" in instance.df.columns
    assert "hr_efficiency" in instance.df.columns
    assert "YearMonth" in instance.df.columns
    assert len(instance.df) == 3
    # Verify sorted by date
    dates = instance.df["Date"].tolist()
    assert dates == sorted(dates)


def test_get_dataframe_returns_copy():
    df = _make_synthetic_df()
    instance = DataLoaderClass.FromDataframe(df)
    copy = instance.GetDataframe()
    copy["Distance"] = 999
    assert (instance.df["Distance"] != 999).all()


# ═══════════════════════════════════════════════
# JoinedDataLoader tests
# ═══════════════════════════════════════════════
from analytics.JoinedDataLoader import JoinedDataLoaderClass


def _load_joined():
    csv_path = Path(__file__).resolve().parent / "data" / "JoinedRunWeather.csv"
    return JoinedDataLoaderClass(csv_path)


def test_weather_columns_created():
    loader = _load_joined()
    expected_cols = [
        "heat_index", "humidity_temp_product", "temp_deviation",
        "dew_point_discomfort", "wind_chill", "gust_ratio",
        "running_stress_score", "cloud_fraction",
    ]
    for col in expected_cols:
        assert col in loader.df.columns, f"Missing weather column: {col}"


def test_get_weather_summary():
    loader = _load_joined()
    summary = loader.GetWeatherSummary()
    assert isinstance(summary, pd.DataFrame)
    assert "YearMonth" in summary.columns
    for col in loader.WeatherCols:
        assert col in summary.columns


def test_get_available_cols():
    loader = _load_joined()
    available = loader.GetAvailableCols()
    assert isinstance(available, list)
    assert len(available) > 0
    for col in available:
        assert col in loader.df.columns
        assert col in loader.AllSelectableCols


def test_correlation_matrix_returns_png():
    loader = _load_joined()
    cols = ["Distance", "Avg HR"]
    png_bytes = loader.CorrelationMatrixPng(cols)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_correlation_matrix_raises_on_few_cols():
    loader = _load_joined()
    with pytest.raises(ValueError):
        loader.CorrelationMatrixPng(["Distance"])


# ═══════════════════════════════════════════════
# App helper function tests
# ═══════════════════════════════════════════════
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import FmtMinutes, BuildInputRow, app as flask_app


def test_fmt_minutes():
    assert FmtMinutes(240.5) == "4h 00m 30s"
    assert FmtMinutes(0.0) == "0h 00m 00s"
    assert FmtMinutes(90.0) == "1h 30m 00s"
    assert FmtMinutes(61.5) == "1h 01m 30s"


def test_build_input_row_returns_dataframe():
    form = {
        "target_finish_time": "4:00",
        "personal_best": "4:30",
        "marathon_weather": "Sunny",
        "injury_severity": "None",
        "injury_count": "0",
        "course_difficulty": "Flat",
        "running_experience_months": "24",
        "resting_heart_rate_bpm": "60",
        "vo2_max": "45",
        "race_month": "4",
    }
    result = BuildInputRow(form)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_build_input_row_default_values():
    form = {}
    result = BuildInputRow(form)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


# ═══════════════════════════════════════════════
# Flask route tests
# ═══════════════════════════════════════════════

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_home_route(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_dashboard_route(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_chart_route_valid(client):
    resp = client.get("/chart/distance_over_time")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"


def test_chart_route_invalid(client):
    resp = client.get("/chart/nonexistent")
    assert resp.status_code == 404


def test_predict_get(client):
    resp = client.get("/predict")
    assert resp.status_code == 200


def test_heatmap_get(client):
    resp = client.get("/heatmap")
    assert resp.status_code == 200


# ═══════════════════════════════════════════════
# Chart generator tests
# ═══════════════════════════════════════════════

def test_distance_chart_returns_bytes():
    ra = _load_analytics()
    chart = DistanceOverTimeChart()
    result = chart.generate(ra.df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_efficiency_chart_returns_bytes():
    ra = _load_analytics()
    chart = EfficiencyOverTimeChart()
    result = chart.generate(ra.df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_efficiency_chart_empty_df():
    chart = EfficiencyOverTimeChart()
    empty_df = pd.DataFrame({"hr_efficiency": [], "Date": []})
    result = chart.generate(empty_df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_weekly_chart_returns_bytes():
    ra = _load_analytics()
    chart = WeeklyLoadVsPaceChart()
    result = chart.generate(ra.df)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ═══════════════════════════════════════════════
# DateHierarchyTree edge cases
# ═══════════════════════════════════════════════

def test_tree_empty_dataframe():
    df = pd.DataFrame({"year": [], "YearMonth": [], "Date": [],
                        "Distance": [], "Avg HR": [], "Avg Pace_sec": [],
                        "Calories": []})
    tree = DateHierarchyTree(df)
    monthly = tree.monthly_summary()
    yearly = tree.yearly_summary()
    assert len(monthly) == 0
    assert len(yearly) == 0


def test_tree_single_run():
    df = pd.DataFrame({
        "year": [2024], "YearMonth": ["2024-03"], "Date": ["2024-03-15"],
        "Distance": [10.0], "Avg HR": [150.0], "Avg Pace_sec": [360.0],
        "Calories": [500.0],
    })
    tree = DateHierarchyTree(df)
    monthly = tree.monthly_summary()
    assert len(monthly) == 1
    assert monthly.iloc[0]["total_distance"] == 10.0
    assert monthly.iloc[0]["run_count"] == 1

    yearly = tree.yearly_summary()
    assert len(yearly) == 1
    assert yearly.iloc[0]["total_distance"] == 10.0


# ═══════════════════════════════════════════════
# HashTable edge case
# ═══════════════════════════════════════════════

def test_hashtable_getitem_raises_keyerror():
    ht = HashTable({"a": 1})
    with pytest.raises(KeyError):
        _ = ht["missing"]


# ═══════════════════════════════════════════════
# Savitzky-Golay filter tests
# ═══════════════════════════════════════════════

def test_savitzky_golay_filter_validation():
    chart = DistanceOverTimeChart()
    with pytest.raises(ValueError):
        chart._savitzky_golay_filter(window_length=4)  # even
    with pytest.raises(ValueError):
        chart._savitzky_golay_filter(window_length=1)  # too small
    with pytest.raises(ValueError):
        chart._savitzky_golay_filter(window_length=5, polyorder=5)  # polyorder >= window


def test_apply_savitzky_golay_preserves_length():
    chart = DistanceOverTimeChart()
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = chart._apply_savitzky_golay_filter(series, window=5, order=2)
    assert len(result) == len(series)