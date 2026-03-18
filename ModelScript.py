import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import joblib
from pathlib import Path
import matplotlib as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

Target          = "actual_finish_time_minutes"
PredictionsPath = Path("data/predictions.csv")
ModelPath       = Path("data/final_model.joblib")

# ── Load data ─────────────────────────────────────────────────────────────
train = pd.read_csv("data/TrainData.csv")
test  = pd.read_csv("data/TestData.csv")

train = train.drop(columns=["medal_outcome", "weekly_mileage_miles"])
test  = test.drop(columns=["weekly_mileage_miles"])
train = train.dropna(subset=[Target])
print("Training rows:", len(train))


def Preprocess(df):
    df = df.copy()

    df["ambition_gap"]    = df["target_finish_time_minutes"] - df["personal_best_minutes"]
    df["race_month"]      = pd.to_datetime(df["marathon_date"]).dt.month
    df["injury_severity"] = df["injury_severity"].fillna("None")

    ordinalColumns = {
        "training_program":  ["Beginner", "Intermediate", "Advanced"],
        "course_difficulty": ["Flat", "Mixed", "Hilly"],
        "injury_severity":   ["None", "Minor", "Moderate", "Severe"],
    }

    for col, categories in ordinalColumns.items():
        encoder = OrdinalEncoder(
            categories=[categories],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        df[col] = encoder.fit_transform(df[[col]])

    df = pd.get_dummies(df, columns=["gender", "marathon_weather"])
    df = df.drop(columns=[c for c in ["runner_id", "marathon_date"] if c in df.columns])

    return df


trainProcessed = Preprocess(train)
testProcessed  = Preprocess(test)

featureCols   = [c for c in trainProcessed.columns if c != Target]
testProcessed = testProcessed.reindex(columns=featureCols, fill_value=0)

trainSorted     = train.sort_values("marathon_date")
trainProcSorted = Preprocess(trainSorted)

Xall  = trainProcSorted[featureCols]
yAll  = trainProcSorted[Target]
Xtest = testProcessed

tscv = TimeSeriesSplit(n_splits=5)

# ── Train + predict only if predictions don't exist ───────────────────────
if not PredictionsPath.exists():

    print("\nCross validation with default parameters")

    baseModel = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )

    maeScores, rmseScores = [], []

    for fold, (trainIdx, valIdx) in enumerate(tscv.split(Xall)):
        Xtr, Xval = Xall.iloc[trainIdx], Xall.iloc[valIdx]
        ytr, yval = yAll.iloc[trainIdx], yAll.iloc[valIdx]

        baseModel.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        preds = baseModel.predict(Xval)

        mae  = mean_absolute_error(yval, preds)
        rmse = root_mean_squared_error(yval, preds)
        maeScores.append(mae)
        rmseScores.append(rmse)
        print(f"Fold {fold+1}: MAE={mae:.2f} | RMSE={rmse:.2f}")

    print("Mean MAE:", np.mean(maeScores))
    print("Mean RMSE:", np.mean(rmseScores))

    print("\nRunning Optuna search (50 trials)")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "rmse",
        }
        foldMae = []
        for trainIdx, valIdx in tscv.split(Xall):
            model = xgb.XGBRegressor(**params)
            model.fit(Xall.iloc[trainIdx], yAll.iloc[trainIdx], verbose=False)
            preds = model.predict(Xall.iloc[valIdx])
            foldMae.append(mean_absolute_error(yAll.iloc[valIdx], preds))
        return np.mean(foldMae)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Best MAE:", study.best_value)
    print("Best params:", study.best_params)

    print("\nTraining final model")

    finalModel = xgb.XGBRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse"
    )
    finalModel.fit(Xall, yAll, verbose=False)

    # ── Save model + assets for prediction page ───────────────────────
    joblib.dump(finalModel,                   ModelPath)
    joblib.dump(featureCols,                  Path("data/feature_cols.joblib"))
    joblib.dump(Xall.median().to_dict(),      Path("data/train_medians.joblib"))

    testPredictions = finalModel.predict(Xtest)
    submission = pd.DataFrame({
        "runner_id":                  test["runner_id"],
        "predicted_finish_time_mins": np.round(testPredictions, 2)
    })
    submission.to_csv(PredictionsPath, index=False)
    print("Saved predictions:", len(submission))
    print(submission.head(10))

else:
    print("\npredictions.csv already exists — skipping training.")
    print("Loading saved model for SHAP plots...")
    finalModel = joblib.load(ModelPath)

# ── SHAP plots always run ─────────────────────────────────────────────────
print("\nGenerating SHAP plots")

explainer  = shap.TreeExplainer(finalModel)
shapValues = explainer.shap_values(Xall)

shap.summary_plot(shapValues, Xall, plot_type="bar", max_display=15, show=True)
shap.summary_plot(shapValues, Xall, max_display=15, show=True)
print("Done")