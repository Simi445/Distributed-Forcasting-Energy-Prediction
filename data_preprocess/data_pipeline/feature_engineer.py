from typing import Dict, List
import pandas as pd
import numpy as np
from ..classes import Engineer, FeatureEngineerConfig
from ..my_logger_pkg import my_logger

class FeatureEngineer(Engineer):
    def feature_engineer_buildings(self, full_df: pd.DataFrame, config: FeatureEngineerConfig) -> Dict[int, pd.DataFrame]:
        weather_cols: List[str] = config.weather_cols or ["air_temperature", "cloud_coverage", "dew_temperature", 
                                            "precip_depth_1_hr","sea_level_pressure", "wind_direction",
                                            "wind_speed"]
        lag_configs: Dict[str, int] = config.lag_configs or {'load_lag_1h': 1, 'load_lag_24h': 24}
        freq: str = config.freq
        interp_limit: int = config.interp_limit
        
        if full_df is None:
            raise ValueError("Merged data not available. Call merge_tables() first.")

        buildings: np.ndarray = full_df['building_id'].unique()
        per_building: Dict[int, pd.DataFrame] = {}

        for b in buildings:
            df_b: pd.DataFrame = full_df[full_df['building_id'] == b].set_index('timestamp').sort_index()
            df_b = self._add_time_features(df_b)
            df_b = df_b.resample(freq).asfreq()

            primary_use_cols: List[str] = [c for c in df_b.columns if 'primary_use' in c]
            for col in primary_use_cols:
                df_b[col] = df_b[col].ffill().bfill().infer_objects(copy=False)

            df_b['meter_reading'] = df_b['meter_reading'].ffill(limit=24).infer_objects(copy=False)
            df_b = df_b.dropna(subset=['meter_reading'])  

            cols: List[str] = [c for c in weather_cols if c in df_b.columns]
            df_b[cols] = df_b[cols].interpolate(limit=interp_limit).ffill().bfill().infer_objects(copy=False)
            
            df_b = self._add_lag_features(df_b, lag_configs)

            df_b = self._add_nonlinear_features(df_b)
            df_b = df_b.dropna()

            per_building[b] = df_b
            my_logger.info(f"Processed building {b} with {len(df_b)} rows.")

        return per_building
    
    def _add_time_features(self, df_b:pd.DataFrame)->pd.DataFrame:
        df_b['hour'] = df_b.index.hour
        df_b['dayofweek'] = df_b.index.dayofweek
        df_b['month'] = df_b.index.month
        df_b['day'] = df_b.index.day
        df_b['is_weekend'] = (df_b.index.dayofweek >= 5).astype(int)
        df_b['is_business_hour'] = df_b['hour'].between(8, 18).astype(int)
        return df_b
    
    def _add_lag_features(self, df_b: pd.DataFrame, lag_configs: Dict[str, int]) -> pd.DataFrame:
        for name, shift in lag_configs.items():
            df_b[name] = df_b['meter_reading'].shift(shift)
        df_b['load_roll_24h_mean'] = df_b['meter_reading'].shift(1).rolling(24).mean()
        df_b['load_roll_24h_std'] = df_b['meter_reading'].shift(1).rolling(24).std()
        df_b['load_roll_168h_mean'] = df_b['meter_reading'].shift(1).rolling(168).mean()
        df_b['load_roll_168h_std'] = df_b['meter_reading'].shift(1).rolling(168).std()
        return df_b
    
    def _add_nonlinear_features(self, df_b: pd.DataFrame) -> pd.DataFrame:
        df_b['hour_x_temp'] = df_b['hour'] * df_b['air_temperature']
        df_b['hour_sin'] = np.sin(2 * np.pi * df_b['hour'] / 24)
        df_b['hour_cos'] = np.cos(2 * np.pi * df_b['hour'] / 24)
        return df_b