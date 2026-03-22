import pandas as pd
from analytics.DataLoader import DataLoaderClass
from analytics.DateHierarchyTree import DateHierarchyTree

class RunningAnalyticsClass(DataLoaderClass):
    """Adds monthly and yearly summary capabilities on top of the base data loader.

    Under the hood, it builds a date hierarchy tree (Year -> Month -> Run)
    at startup and uses recursive traversal for aggregations instead of
    pandas groupby.
    """

    def __init__(self, Filepath):
        """Load data from CSV and build the date tree for fast aggregation."""
        super().__init__(Filepath)
        self._date_tree = DateHierarchyTree(self.df)

    @classmethod
    def FromDataframe(cls, df: pd.DataFrame):
        """Build from an in-memory DataFrame, including the date tree."""
        instance = super().FromDataframe(df)
        instance._date_tree = DateHierarchyTree(instance.df)
        return instance

    def MonthlySummary(self) -> pd.DataFrame:
        """Get a breakdown of distance, pace, heart rate, and calories by month.

        Returns one row per month with totals and averages across all runs.
        """
        return self._date_tree.monthly_summary()

    def YearlySummary(self) -> pd.DataFrame:
        """Same as MonthlySummary but rolled up to the year level."""
        return self._date_tree.yearly_summary()
