import mpld3
import matplotlib.pyplot as plt
from ..classes import Plot
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
from ..my_logger_pkg import my_logger

class PlotHelper(Plot):
    def __init__(self, building_counts: Dict[str, int]):
        self.building_counts = building_counts
        self.categories_selected = list(self.building_counts.keys())
        self.ids = list(self.building_counts.values())
        my_logger.info(f"PlotHelper initialized with {len(self.ids)} buildings.")

    def plot_acf(self, acf24_per_building: Dict[int, float]) -> str:
        my_logger.info("Starting ACF plot generation")
        acfs: Dict[int, float] = {}
        for b_id in self.ids:
            acfs[b_id] = acf24_per_building.get(b_id, 0)

        fig, ax = plt.subplots()
        ax.bar(range(len(self.ids)), [acfs.get(b, 0) for b in self.ids], 
               tick_label=[f'{cat}\n(ID: {b})' for cat, b in zip(self.categories_selected, self.ids)], align='center')
        ax.set_ylabel('ACF (lag 24)')
        ax.set_title('ACF values for selected buildings')

        html:str = mpld3.fig_to_html(fig)
        return html

    def plot_energy(self, per_building: Dict[int, pd.DataFrame]) -> str:
        my_logger.info("Starting energy consumption plot generation")
        num_buildings = len(self.ids)
        fig, ax = plt.subplots(num_buildings, 1, figsize=(10, 8 * num_buildings // 2))

        for i, (category, b) in enumerate(zip(self.categories_selected, self.ids)):
            if b in per_building and 'meter_reading' in per_building[b].columns:
                s = per_building[b]['meter_reading']
                ax[i].plot(s.index, s.values)
                ax[i].set_title(f'Electricity consumption for {category} (Building {b})')
                my_logger.debug(f"Plotted data for building {b} ({category})")
            else:
                ax[i].text(0.5, 0.5, f'No data for {category} (Building {b})', ha='center', va='center', transform=ax[i].transAxes)
                my_logger.warning(f"No data available for building {b} ({category})")

        plt.tight_layout()

        html:str = mpld3.fig_to_html(fig)
        return html