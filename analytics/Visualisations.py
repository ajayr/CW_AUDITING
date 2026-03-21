from analytics.RunningAnalytics import RunningAnalyticsClass
from analytics.chart_generators import (
    DistanceOverTimeChart,
    EfficiencyOverTimeChart,
    WeeklyLoadVsPaceChart,
)


class VisualisationDashboardClass(RunningAnalyticsClass):

    def __init__(self, Filepath):
        super().__init__(Filepath)
        self._distance_chart = DistanceOverTimeChart()
        self._efficiency_chart = EfficiencyOverTimeChart()
        self._weekly_chart = WeeklyLoadVsPaceChart()

    @classmethod
    def FromDataframe(cls, df):
        instance = super().FromDataframe(df)
        instance._distance_chart = DistanceOverTimeChart()
        instance._efficiency_chart = EfficiencyOverTimeChart()
        instance._weekly_chart = WeeklyLoadVsPaceChart()
        return instance

    def DistanceOverTime(self) -> bytes:
        return self._distance_chart.generate(self.df)

    def EfficiencyOverTime(self, start_date=None, end_date=None) -> bytes:
        return self._efficiency_chart.generate(
            self.df, start_date=start_date, end_date=end_date
        )

    def WeeklyLoadVsPace(self) -> bytes:
        return self._weekly_chart.generate(self.df)
