import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from analytics.RunningAnalytics import RunningAnalyticsClass
from analytics.JoinedDataLoader import JoinedDataLoaderClass
import pandas as pd


class VisualisationDashboardClass(RunningAnalyticsClass):

    FigSize = (12, 5)
    Style   = "seaborn-v0_8-whitegrid"

    def _NewFig(self):
        plt.style.use(self.Style)
        return plt.subplots(figsize=self.FigSize)

    def _ToPng(self, fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        plt.close(fig)
        return buf.read()

    def _SavitzkyGolayFilter(self, window_length =31, polyorder = 3, deriv = 0 ):
        if window_length <= 2:
            raise ValueError("window_length must be odd")
        if polyorder>window_length:
            raise ValueError("polyorder must be smaller than window_length")

        half = (window_length-1) // 2
        x = np.arange(-half, half +1)

        VandermondeMatrix = np.vander(x, polyorder+1, increasing = True)

        if deriv == 0:
            target = np.zeros(polyorder+1)
            target[0] = 1
        else:
            target = np.zeros(polyorder+1)
            target[deriv] = np.math.factorial(deriv)

        kernelcoeff = np.linalg.pinv(VandermondeMatrix) @ target
        return kernelcoeff

    def _ApplySavitzkyGolayFilter(self, series, window = 31, order = 3):
        kernel = self._SavitzkyGolayFilter(window, order)
        smoothedsignal = np.convolve(series.values, kernel[::-1], mode='same')
        return pd.Series(smoothedsignal, index=series.index)



    def DistanceOverTime(self) -> bytes:
        df = self.df.dropna(subset=["Distance", "Date"]).copy()
        df["Rolling"] = df["Distance"].rolling(window=4, min_periods=1).mean()
        sg_window = 61
        sg_order = 3
        try:
            df["savgol_distance"] = self._ApplySavitzkyGolayFilter(
                df["Rolling"],
                window=sg_window,
                order=sg_order
            )
        except Exception as e:
            print(f"Savitzky-Golay failed: {e}")
            df["savgol_distance"] = df["Rolling"]  # fallback

        df = df.reset_index()


        fig, ax = self._NewFig()
        ax.scatter(df["Date"], df["Distance"],
                   s=15, alpha=0.5, color="royalblue", label="Run Distance")
        ax.plot(df["Date"], df["Rolling"],
                color="tomato", linewidth=1.8, label="4-run Rolling Avg")

        ax.plot(
            df["Date"], df["savgol_distance"],
            color="#c0392b",  # strong red to stand out
            linewidth=2.8,
            linestyle="--",
            label=f"Savitzky–Golay (win={sg_window}, order={sg_order})",
            zorder=4,
            alpha=0.95
        )

        ax.set(title="Distance Over Time", xlabel="Date", ylabel="Distance (km)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend()
        return self._ToPng(fig)

    def EfficiencyOverTime(self) -> bytes:
        df = self.df.dropna(subset=["hr_efficiency", "Date"]).copy()

        df["efficiency"] = 1 / df["hr_efficiency"]

        def classify(title: str) -> str:
            t = str(title).lower()
            if "interval" in t or "fartlek" in t:  return "Intervals"
            if "threshold" in t:                    return "Threshold"
            if "long" in t:                         return "Long Run"
            if "easy" in t or "steady" in t:        return "Easy"
            return "Other"

        df["run_type"] = df["Title"].apply(classify)

        df = df.sort_values("Date").set_index("Date")
        df["rolling_eff"] = df["efficiency"].rolling("30D", min_periods=1).mean()
        df = df.reset_index()

        sg_window = 61
        sg_order = 3
        try:
            df["savgol_eff"] = self._ApplySavitzkyGolayFilter(
                df["efficiency"],
                window=sg_window,
                order=sg_order
            )
        except Exception as e:
            print(f"Savitzky-Golay failed: {e}")
            df["savgol_eff"] = df["rolling_eff"]  # fallback

        df = df.reset_index()


        colours = {
            "Easy":      "#4a90d9",
            "Threshold": "#e94560",
            "Intervals": "#f39c12",
            "Long Run":  "#27ae60",
            "Other":     "#a0aec0",
        }

        fig, ax = self._NewFig()


        fig.set_size_inches(14, 6)

        for runType, colour in colours.items():
            subset = df[df["run_type"] == runType]
            if subset.empty:
                continue
            ax.scatter(subset["Date"], subset["efficiency"],
                       s=18, alpha=0.45, color=colour,
                       label=runType, zorder=2)

        ax.plot(df["Date"], df["rolling_eff"],
                color="#2c3e50", linewidth=2,
                label="30-day Rolling Avg", zorder=3)
        ax.plot(
            df["Date"], df["savgol_eff"],
            color="#c0392b",  # strong red to stand out
            linewidth=2.8,
            linestyle="--",
            label=f"Savitzky–Golay (win={sg_window}, order={sg_order})",
            zorder=4,
            alpha=0.95
        )

        ax.set(
            title="Running Efficiency (Speed/HR) Over Time",
            xlabel="Date",
            ylabel="Efficiency (higher = better)",
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        return self._ToPng(fig)

    def WeeklyLoadVsPace(self) -> bytes:
        df = self.df.dropna(subset=["week_start", "Avg Pace_sec"]).copy()

        df["load_metric"] = (
            df["Training Stress Score®"]
            .replace(0, np.nan)
            .fillna(df["duration_min"])
        )
        weekly = (
            df.groupby("week_start")
            .agg(
                total_load=("load_metric",   "sum"),
                best_pace= ("Avg Pace_sec",  "min"),
                year=      ("year",          "first"),
            )
            .reset_index()
            .dropna(subset=["total_load", "best_pace"])
        )

        weekly = weekly[weekly["total_load"] > 5]

        if weekly.empty:
            fig, ax = self._NewFig()
            ax.text(0.5, 0.5, "Not enough data to plot",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title("Weekly Training Load vs Best Pace")
            return self._ToPng(fig)

        years = sorted(weekly["year"].unique())
        norm  = plt.Normalize(vmin=min(years), vmax=max(years))
        cmap  = plt.cm.plasma

        fig, ax = self._NewFig()
        fig.set_size_inches(13, 7)

        scatter = ax.scatter(
            weekly["total_load"],
            weekly["best_pace"],
            c=weekly["year"],
            cmap="plasma",
            norm=norm,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )

        x      = weekly["total_load"].values
        y      = weekly["best_pace"].values
        coeffs = np.polyfit(x, y, deg=2)
        poly   = np.poly1d(coeffs)
        xSmooth = np.linspace(x.min(), x.max(), 300)
        ax.plot(xSmooth, poly(xSmooth),
                color="#e94560", linewidth=2.2,
                linestyle="--", label="Regression curve", zorder=3)

        def SecToPace(sec, _):
            sec = int(sec)
            return f"{sec // 60}:{sec % 60:02d}"

        ax.yaxis.set_major_formatter(plt.FuncFormatter(SecToPace))
        ax.invert_yaxis()

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Year", fontsize=9)
        cbar.set_ticks(years[::2])

        ax.set(
            title="Weekly Training Load vs Best Pace",
            xlabel="Weekly Load  (TSS or duration mins)",
            ylabel="Best Avg Pace that week",
        )
        ax.axvspan(150, 350, alpha=0.08, color="green", label="Optimal load zone")
        weekly = weekly[weekly["best_pace"].between(300, 800)]
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._ToPng(fig)