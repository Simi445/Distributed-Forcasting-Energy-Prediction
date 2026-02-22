import pandas as pd
from my_logger_pkg import my_logger
from typing import Dict, Tuple
from functools import reduce
from classes import SeasonConfig, BaselinePredictor
from seasonality import SeasonMaskStrategy

from auxiliary import BaseLineDataOrganizer

class MeanImpl(BaselinePredictor):

    def __init__(self, strategy=None):
        super().__init__(
            data_preparator=BaseLineDataOrganizer(),
            strategy=strategy or SeasonMaskStrategy()
        )
        my_logger.info(f"Initializing mean model.")
    
    def train(self, season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tt_res:Dict = self.get_train_test()
        final_res: Dict = {}
        for build_id, (y, y_test) in tt_res.items():
            dataframes = [y.shift(i).loc[y_test.index] for i in range(24, 169, 24)]
            y_hat = reduce(lambda x, z: x + z, dataframes) / len(dataframes)
            y_pred, y_test = self._apply_seasonal_filter(y_hat, y_test, season)
            final_res[build_id] = (y_pred, y_test)
        return final_res
    
