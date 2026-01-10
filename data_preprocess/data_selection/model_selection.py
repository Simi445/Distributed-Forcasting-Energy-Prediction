from typing import Dict, List, Tuple
import statistics
import pandas as pd
from ..classes import ModelBuildings, FilterConfig
from .filter import FilterBuildings
from ..my_logger_pkg import my_logger

class ModelBSelection(ModelBuildings):
    def __init__(self, filter_fn=FilterBuildings().filter_buildings):
        self.filter_fn = filter_fn
        my_logger.info("ModelBSelection initialized with custom filter function")

    def select_final_buildings(self, per_building: Dict[int, pd.DataFrame], median_buildings: Dict[str, Tuple[int, float]], k: int = 3) -> Dict[str, int]:
        my_logger.info(f"Starting final building selection with k={k}")
        category_acfs: List[Tuple[str, int, float]] = [(c, b, acf) for c, (b, acf) in median_buildings.items()]

        global_median: float = statistics.median(acf for _, _, acf in category_acfs)
        my_logger.info(f"Global median ACF: {global_median}")

        ranked_neighbors: List[Tuple[str, int, float]] = sorted(category_acfs, key=lambda x: x[2] - global_median)
        my_logger.debug(f"Ranked buildings by ACF deviation: {[(c, b, acf) for c, b, acf in ranked_neighbors]}")

        selected: Dict[str, int] = {c: b for c, b, _ in ranked_neighbors[:k]}
        my_logger.info(f"Top {k} selected buildings: {selected}")

        final = self.filter_fn(build_dict=selected, per_building=per_building, config=FilterConfig(k=k))
        my_logger.info(f"Final filtered buildings: {final}")

        result = {c: b for (_, c, b) in final}
        my_logger.info(f"Returning final selection: {result}")
        return result