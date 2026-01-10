import pandas as pd
from .classes import SeasonalityStrategy,Mask
from auxiliary import SeasonMask

class NoSeasonalityStrategy(SeasonalityStrategy):
    def apply(self, train_data: pd.DataFrame, fold_key: str) -> pd.DataFrame:
        return train_data

class SeasonMaskStrategy(SeasonalityStrategy):
    def apply(self, train_data: pd.DataFrame, fold_key: str) -> pd.DataFrame:
        mask = SeasonMask(Mask.get_season_mask(train_data, fold_key))
        return train_data[mask]