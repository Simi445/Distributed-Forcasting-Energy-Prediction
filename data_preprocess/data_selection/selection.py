from ..my_logger_pkg import my_logger
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
from ..classes import Selection, ACFConfig

class EnhancedSelection(Selection):
    def category_buildings(self, per_building: Dict[int, pd.DataFrame]) -> Dict[int, str]:
        my_logger.info("Starting building categorization")
        building_type: Dict[int, str] = {}
        for b, df_b in per_building.items():
            df_b = df_b.filter(regex='primary_use').head(1)
            for _, row in df_b.iterrows():
                for r, v in row.items():
                    if v == True:
                        building_type[b] = r
        my_logger.info(f"Categorized {len(building_type)} buildings and their types")
        return building_type
    

    def calculate_lag_acf(self, building_type: Dict[int, str],
                          per_building: Dict[int, pd.DataFrame],
                          acf_config: Optional[ACFConfig] = None) -> Tuple[Dict[str,Any], Dict[int,float]]:
        my_logger.info("Starting ACF calculation for lag 24")
        
        if not building_type:
            my_logger.warning("No building types provided for ACF calculation")
            return {}, {}
        if not per_building:
            my_logger.warning("No building data provided for ACF calculation")
            return {}, {}
        
        if acf_config is None:
            acf_config = ACFConfig()
        
        months: List[int] = [m for m in range(1, 7)]
        acf24_per_type: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        acf24_per_building: Dict[int,float] = defaultdict(float)
        processed = 0
        skipped = 0
        
        for b, category in building_type.items():
            df_b = per_building[b]
            df_b = df_b[df_b.index.month.isin(months)]
            s = df_b['meter_reading'].dropna()
            if len(s) < acf_config.hours * acf_config.nr_days:
                skipped += 1
                continue
            
            if s.isna().any() or np.isinf(s).any() or (s.std() == 0):
                my_logger.debug(f"Skipping building {b} ({category}): has NaN, Inf, or zero variance")
                skipped += 1
                continue
                
            acf_val = acf(s, nlags=24, fft=True)[24]
            
            if np.isnan(acf_val):
                my_logger.debug(f"Skipping building {b} ({category}): ACF calculation resulted in NaN")
                skipped += 1
                continue
            
            acf24_per_type[category].append((b, acf_val))
            acf24_per_building[b] = acf_val
            processed += 1
        
        my_logger.info(f"Calculated ACF for {processed} buildings, skipped {skipped} due to insufficient data")
        my_logger.info(f"ACF results for {len(acf24_per_type)} categories")
        return (acf24_per_type, acf24_per_building)
    

    def median_per_type(self, acf24_per_type: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Tuple[int, float]]:
        my_logger.info("Calculating median buildings per type")
        
        if not acf24_per_type:
            my_logger.warning("No ACF data provided for median calculation")
            return {}
        
        median_buildings: Dict[str, Tuple[int, float]] = {}
        for b_type, vals in acf24_per_type.items():
            if not vals:
                my_logger.warning(f"No values for building type {b_type}")
                continue
                
            acfs = np.array([a for _, a in vals])
            med = np.median(acfs)

            b_med, a_med = min(vals, key=lambda x: abs(x[1] - med))
            median_buildings[b_type] = (b_med, a_med)
            my_logger.debug(f"Median for {b_type}: building {b_med}, ACF {a_med}")

        median_buildings = dict(sorted(median_buildings.items(), key=lambda item: item[1][1]))
        my_logger.info(f"Median buildings calculated and sorted for {len(median_buildings)} types")
        return median_buildings