import pandas as pd
from my_logger_pkg import my_logger
from typing import Optional, Dict, Tuple
from classes import Organizer, BaselinePredictor, SeasonalityStrategy

class NaiveImpl(BaselinePredictor):

   def __init__(self, organizer:Organizer, strategy: SeasonalityStrategy):
        super().__init__(
            data_preparator=organizer,
            strategy=strategy
        )
        my_logger.info(f"Initializing naive model.")
    
   def train(self, season: str) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        tt_res:Dict = self.get_train_test()
        final_res: Dict = {}
        for build_id, (y, y_test) in tt_res.items():
            y_hat = y.shift(24).loc[y_test.index]
            y_pred, y_test = self._apply_seasonal_filter(y_hat, y_test, season)
            final_res[build_id] = (y_pred, y_test)
        return final_res
