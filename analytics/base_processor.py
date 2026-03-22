from abc import ABC, abstractmethod
import pandas as pd


class BaseDataProcessor(ABC):
    """Every data processor in this project inherits from here.

    It's a simple contract -- if you load or transform data, you need
    to implement a process() method. That way the rest of the codebase
    can treat all processors the same way.
    """

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a raw DataFrame and return a cleaned, enriched version of it."""
        ...
