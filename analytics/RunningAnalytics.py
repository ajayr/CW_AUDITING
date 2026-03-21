import pandas as pd
from analytics.DataLoader import DataLoaderClass
from analytics.DateHierarchyTree import DateHierarchyTree

class RunningAnalyticsClass(DataLoaderClass):

    def __init__(self, Filepath):
        super().__init__(Filepath)
        self._date_tree = DateHierarchyTree(self.df)

    @classmethod
    def FromDataframe(cls, df: pd.DataFrame):
        instance = super().FromDataframe(df)
        instance._date_tree = DateHierarchyTree(instance.df)
        return instance

    def MonthlySummary(self) -> pd.DataFrame:
        return self._date_tree.monthly_summary()

    def YearlySummary(self) -> pd.DataFrame:
        return self._date_tree.yearly_summary()