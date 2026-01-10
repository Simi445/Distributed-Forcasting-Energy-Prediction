import pandas as pd
from typing import Dict, List, Tuple, Union
import heapq
from ..classes import Loader, FilterConfig, Filter
from ..my_logger_pkg import my_logger

class FilterBuildings(Filter):
    def quality_check(self, df_b: pd.DataFrame, freq: str = 'h') -> Dict[str, Union[float, pd.Timedelta]]:
        my_logger.debug("Performing quality check on building data")
        idx = pd.DatetimeIndex(pd.to_datetime(df_b.index)).sort_values()
        u = idx.unique()
        start, end = u.min(), u.max()
        duration_days: float = (end - start) / pd.Timedelta(days=1)

        expected = pd.date_range(start=start, end=end, freq=freq)
        expected_n: int = len(expected)
        actual_n: int = len(u)

        missing_ratio: float = (1 - (actual_n / expected_n))

        max_gap: pd.Timedelta = pd.Timedelta(0)
        if actual_n >= 2:
            max_gap = (u[1:] - u[:-1]).max()
        
        result = {
            "max_gap": max_gap,
            "missing_ratio": missing_ratio,
            "duration_days": duration_days
        }
        my_logger.debug(f"Quality check results: {result}")
        return result
    
    def filter_buildings(self, build_dict: Dict[str, int], per_building: Dict[int, pd.DataFrame], config: FilterConfig) -> List[Tuple[Tuple[float, pd.Timedelta, float], str, int]]:
        my_logger.info(f"Starting building filtering with config: max_gap={config.max_gap_th}, miss_ratio={config.miss_ratio_th}, days_min={config.days_th_min}, k={config.k}")
        res: List[Tuple[Tuple[float, pd.Timedelta, float], str, int]] = []
        passed = 0
        failed = 0
        for c, b in build_dict.items():
            df_b = per_building[b]
            dict_res: Dict[str, Union[float, pd.Timedelta]] = self.quality_check(df_b=df_b)
            max_gap: pd.Timedelta = dict_res['max_gap']
            miss_ratio: float = dict_res['missing_ratio']
            duration_days: float = dict_res['duration_days']
            if max_gap < config.max_gap_th and miss_ratio < config.miss_ratio_th and duration_days >= config.days_th_min:
                key = (miss_ratio, max_gap, -duration_days)
                heapq.heappush(res, (key, c, b))
                passed += 1
            else:
                failed += 1
                my_logger.debug(f"Building {b} ({c}) failed filter: gap={max_gap}, miss={miss_ratio}, days={duration_days}")
        
        top_k = heapq.nsmallest(config.k, res)
        my_logger.info(f"Filtered {passed} buildings passed, {failed} failed. Selected top {len(top_k)}: {[(c, b) for _, c, b in top_k]}")
        return top_k