import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from analytics.RunningAnalytics import RunningAnalyticsClass
from analytics.JoinedDataLoader import JoinedDataLoaderClass
from analytics.mergesort import mergesort, mergesort_dataframe
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

    def _SavitzkyGolayFilter(self, window_length=31, polyorder=3, deriv=0):
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if window_length < 3:
            raise ValueError("window_length must be at least 3")
        if polyorder >= window_length:
            raise ValueError("polyorder must be smaller than window_length")
        if deriv > polyorder:
            raise ValueError("deriv must be <= polyorder")

        half = (window_length - 1) // 2
        x = np.arange(-half, half + 1, dtype=float)

        vandermonde = np.vander(x, polyorder + 1, increasing=True)

        target = np.zeros(polyorder + 1)
        if deriv == 0:
            target[0] = 1.0
        else:
            target[deriv] = np.math.factorial(deriv)

        # Solve V^T c = target for c, where c has length window_length
        kernelcoeff = np.linalg.pinv(vandermonde.T) @ target
        return kernelcoeff

    def _ApplySavitzkyGolayFilter(self, series, window=31, order=3):
        values = np.asarray(series.values, dtype=float)
        n = len(values)

        if n == 0:
            return pd.Series(values, index=series.index)

        if window > n:
            window = n if n % 2 == 1 else n - 1

        if window < 3:
            return pd.Series(values, index=series.index)

        if window % 2 == 0:
            window -= 1

        order = min(order, window - 1)

        kernel = self._SavitzkyGolayFilter(window, order)

        half = window // 2
        padded = np.pad(values, (half, half), mode="edge")
        smoothedsignal = np.convolve(padded, kernel[::-1], mode="valid")

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

    def EfficiencyOverTime(self, start_date=None, end_date=None) -> bytes:
        df = self.df.dropna(subset=["hr_efficiency", "Date"]).copy()

        if df.empty:
            fig, ax = self._NewFig()
            ax.text(
                0.5, 0.5,
                "No efficiency data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14
            )
            ax.set_title("Running Efficiency Over Time")
            return self._ToPng(fig)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = mergesort_dataframe(df.dropna(subset=["Date"]), by="Date")

        if start_date:
            start = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start):
                df = df[df["Date"] >= start]

        if end_date:
            end = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(end):
                df = df[df["Date"] <= end]

        if df.empty:
            fig, ax = self._NewFig()
            ax.text(
                0.5, 0.5,
                "No runs found for the selected date range",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14
            )
            ax.set_title("Running Efficiency Over Time")
            return self._ToPng(fig)

        # Higher efficiency = better
        df["efficiency"] = 1.0 / df["hr_efficiency"]

        # Build a daily time series so smoothing follows time properly
        daily = (
            df.set_index("Date")["efficiency"]
            .resample("D")
            .mean()
            .interpolate(method="time")
        )

        n = len(daily)

        # Adaptive odd window size
        sg_window = min(21, n if n % 2 == 1 else n - 1)
        sg_order = 3

        if sg_window < 5:
            smooth = daily.copy()
        else:
            try:
                smooth = self._ApplySavitzkyGolayFilter(
                    daily,
                    window=sg_window,
                    order=min(sg_order, sg_window - 2)
                )
            except Exception:
                smooth = daily.copy()

        fig, ax = self._NewFig()
        fig.set_size_inches(14, 6)

        # Actual workout points
        ax.scatter(
            df["Date"],
            df["efficiency"],
            s=20,
            alpha=0.35,
            color="#0000FF",
            edgecolors="none",
            label="Actual runs",
            zorder=2
        )

        # Smoothed trend line
        ax.plot(
            daily.index,
            smooth.values,
            color="#c0392b",
            linewidth=2.8,
            label="Trend",
            zorder=3
        )

        ax.set_title("Running Efficiency Over Time", fontsize=16, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Efficiency (higher = better)")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        ax.legend(frameon=False)
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

        years = mergesort(list(weekly["year"].unique()))
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