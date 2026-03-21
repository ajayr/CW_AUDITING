from flask import Flask, render_template, send_file, abort, request, session
import io
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from analytics.Visualisations import VisualisationDashboardClass
from analytics.JoinedDataLoader import JoinedDataLoaderClass
from analytics.hashtable import HashTable

app = Flask(__name__)
app.secret_key = "running-analytics-secret"

@app.context_processor
def inject_request():
    return dict(request=request)

BaseDir      = Path(__file__).resolve().parent
dashboard    = VisualisationDashboardClass(BaseDir / "data" / "GarminFullRunning.csv")
joinedLoader = JoinedDataLoaderClass(BaseDir / "data" / "JoinedRunWeather.csv")

predictModel = joblib.load(BaseDir / "data" / "final_model.joblib")
featureCols  = joblib.load(BaseDir / "data" / "feature_cols.joblib")
trainMedians = joblib.load(BaseDir / "data" / "train_medians.joblib")

graphs = {
    "distance_over_time":   ("Distance Over Time",           dashboard.DistanceOverTime),
    "efficiency_over_time": ("Running Efficiency Over Time", dashboard.EfficiencyOverTime),
    "weekly_load_vs_pace":  ("Weekly Training Load vs Pace", dashboard.WeeklyLoadVsPace),
}

PerformanceCols = [
    "Distance", "Avg Pace_sec", "Avg HR", "Max HR", "Calories",
    "duration_min", "speed_kmh", "hr_efficiency", "Aerobic TE",
    "Avg Run Cadence", "Avg Stride Length", "Avg Vertical Ratio",
    "Avg Vertical Oscillation", "Avg Ground Contact Time",
    "Total Ascent", "Total Descent", "Avg Power",
]
WeatherRawCols = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
]
WeatherDerivedCols = [
    "heat_index", "humidity_temp_product", "temp_deviation",
    "dew_point_discomfort", "wind_chill", "gust_ratio",
    "running_stress_score", "cloud_fraction",
]

InjuryMap = HashTable({"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3})
CourseMap = HashTable({"Flat": 0, "Mixed": 1, "Hilly": 2})

def _FilterExisting(cols):
    available = joinedLoader.GetAvailableCols()
    return [c for c in cols if c in available]

def BuildInputRow(form) -> pd.DataFrame:
    targetStr  = form.get("target_finish_time", "4:00")
    h, m       = map(int, targetStr.split(":"))
    targetMins = h * 60 + m

    personalBestStr = form.get("personal_best", "4:30")
    ph, pm          = map(int, personalBestStr.split(":"))
    personalBest    = ph * 60 + pm

    weather = form.get("marathon_weather", "Sunny")

    row = dict(trainMedians)

    row["target_finish_time_minutes"] = targetMins
    row["personal_best_minutes"]      = personalBest
    row["ambition_gap"]               = targetMins - personalBest
    row["injury_severity"]            = InjuryMap.get(form.get("injury_severity", "None"), 0)
    row["injury_count"]               = int(form.get("injury_count", 0))
    row["course_difficulty"]          = CourseMap.get(form.get("course_difficulty", "Flat"), 0)
    row["running_experience_months"]  = float(form.get("running_experience_months", 24))
    row["resting_heart_rate_bpm"]     = float(form.get("resting_heart_rate_bpm", 60))
    row["vo2_max"]                    = float(form.get("vo2_max", 45))
    row["race_month"]                 = int(form.get("race_month", 4))

    for col in featureCols:
        if col.startswith("marathon_weather_"):
            row[col] = 0
    row[f"marathon_weather_{weather}"] = 1

    df = pd.DataFrame([row])
    df = df.reindex(columns=featureCols, fill_value=0)
    return df

def FmtMinutes(mins: float) -> str:
    h = int(mins) // 60
    m = int(mins) % 60
    s = int(round((mins - int(mins)) * 60))
    return f"{h}h {m:02d}m {s:02d}s"


@app.route("/")
def home():
    return render_template("home.html", active_tab="home")


@app.route("/dashboard")
def index():
    monthly   = dashboard.MonthlySummary().to_dict(orient="records")
    yearly    = dashboard.YearlySummary().to_dict(orient="records")
    chartMeta = {k: v[0] for k, v in graphs.items()}
    return render_template("dashboard.html",
                           active_tab="dashboard",
                           chart_meta=chartMeta,
                           monthly=monthly,
                           yearly=yearly)


@app.route("/chart/<chart_name>")
def get_chart(chart_name: str):
    if chart_name not in graphs:
        abort(404)

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if chart_name == "efficiency_over_time":
        png_bytes = graphs[chart_name][1](
            start_date=start_date,
            end_date=end_date
        )
    else:
        png_bytes = graphs[chart_name][1]()

    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.route("/heatmap", methods=["GET", "POST"])
def heatmap():
    error        = None
    selectedCols = []
    showChart    = False

    if request.method == "POST":
        selectedCols = request.form.getlist("columns")
        if len(selectedCols) < 2:
            error = "Please select at least 2 variables to generate a matrix."
        else:
            showChart = True

    return render_template("heatmap.html",
                           active_tab="heatmap",
                           performance_cols=_FilterExisting(PerformanceCols),
                           weather_raw_cols=_FilterExisting(WeatherRawCols),
                           weather_derived_cols=_FilterExisting(WeatherDerivedCols),
                           selected_cols=selectedCols,
                           show_chart=showChart,
                           error=error)


@app.route("/heatmap/chart", methods=["POST"])
def heatmap_chart():
    selectedCols              = request.form.getlist("columns")
    currentTheme              = session.get("theme_index", 0)
    session["theme_index"]    = (currentTheme + 1) % len(joinedLoader.Themes)

    try:
        pngBytes = joinedLoader.CorrelationMatrixPng(selectedCols, ThemeIndex=currentTheme)
        return send_file(io.BytesIO(pngBytes), mimetype="image/png")
    except ValueError:
        abort(400)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error      = None

    if request.method == "POST":
        try:
            inputDf    = BuildInputRow(request.form)
            predMins   = predictModel.predict(inputDf)[0]
            prediction = FmtMinutes(predMins)
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template("predict.html",
                           active_tab="predict",
                           prediction=prediction,
                           error=error)


if __name__ == "__main__":
    app.run(port=8000, debug=True)