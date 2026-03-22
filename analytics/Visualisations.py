from analytics.RunningAnalytics import RunningAnalyticsClass
from analytics.chart_generators import (
    DistanceOverTimeChart,
    EfficiencyOverTimeChart,
    WeeklyLoadVsPaceChart,
)


class VisualisationDashboardClass(RunningAnalyticsClass):
    """The main dashboard class that ties data loading and chart generation together.

    It inherits all the data loading and summary capabilities, and adds three
    chart types. Each chart is handled by its own generator class — this class
    just wires them up and delegates.
    """

    def __init__(self, Filepath):
        """Load the data and set up the three chart generators."""
        super().__init__(Filepath)
        self._distance_chart = DistanceOverTimeChart()
        self._efficiency_chart = EfficiencyOverTimeChart()
        self._weekly_chart = WeeklyLoadVsPaceChart()

    @classmethod
    def FromDataframe(cls, df):
        """Build from an in-memory DataFrame, including chart generators."""
        instance = super().FromDataframe(df)
        instance._distance_chart = DistanceOverTimeChart()
        instance._efficiency_chart = EfficiencyOverTimeChart()
        instance._weekly_chart = WeeklyLoadVsPaceChart()
        return instance

    def DistanceOverTime(self) -> bytes:
        """Plot how far each run was over time, with rolling average and trend line."""
        return self._distance_chart.generate(self.df)

    def EfficiencyOverTime(self, start_date=None, end_date=None) -> bytes:
        """Plot running efficiency over time, optionally filtered to a date range."""
        return self._efficiency_chart.generate(
            self.df, start_date=start_date, end_date=end_date
        )

    def WeeklyLoadVsPace(self) -> bytes:
        """Scatter plot showing whether heavier training weeks lead to faster paces."""
        return self._weekly_chart.generate(self.df)
