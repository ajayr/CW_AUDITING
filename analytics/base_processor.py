from abc import ABC, abstractmethod
import pandas as pd


class BaseDataProcessor(ABC):
    """Abstract base class for all data processors in the pipeline."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into processed form."""
        ...
