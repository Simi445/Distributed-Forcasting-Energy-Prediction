import mpld3
import matplotlib.pyplot as plt
from .base_plot import Plot, ModelPlot
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
from my_logger_pkg import my_logger

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

        values = [acfs.get(b, 0) for b in self.ids]
        labels = [f'{cat.replace("primary_use_", "").replace("_", " ")}\nID {b}' for cat, b in zip(self.categories_selected, self.ids)]

        fig, ax = plt.subplots(figsize=(12, 4.5))
        bars = ax.bar(range(len(self.ids)), values, tick_label=labels, align='center', color="#2563eb")
        ax.set_ylabel('ACF (lag 24)')
        ax.set_title('ACF values for selected buildings')
        ax.set_ylim(0, max(values) * 1.18 if values else 1)
        ax.grid(axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()

        html:str = mpld3.fig_to_html(fig)
        plt.close(fig)
        return html

    def plot_energy(self, per_building: Dict[int, pd.DataFrame]) -> str:
        my_logger.info("Starting energy consumption plot generation")
        num_buildings = len(self.ids)
        fig, ax = plt.subplots(num_buildings, 1, figsize=(14, 8 * num_buildings // 2))

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
        plt.close(fig)
        return html

    def plot_energy_for_building(self, per_building: Dict[int, pd.DataFrame], category: str, building_id: int) -> str:
        my_logger.info(f"Starting energy consumption plot for building {building_id}")
        fig, ax = plt.subplots(figsize=(12, 5))
        display_category = category.replace("primary_use_", "").replace("_", " ")

        if building_id in per_building and 'meter_reading' in per_building[building_id].columns:
            series = per_building[building_id]['meter_reading']
            ax.plot(series.index, series.values, color="#2563eb", linewidth=0.9)
            ax.set_title(f'Electricity consumption for {display_category} (Building {building_id})')
            ax.set_ylabel('Meter reading')
            ax.set_xlabel('Timestamp')
            ax.grid(alpha=0.25)
            ax.margins(x=0.01, y=0.08)
            fig.autofmt_xdate(rotation=25, ha='right')
            my_logger.debug(f"Plotted data for building {building_id} ({category})")
        else:
            ax.text(
                0.5,
                0.5,
                f'No data for {display_category} (Building {building_id})',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            my_logger.warning(f"No data available for building {building_id} ({category})")

        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()

        html: str = mpld3.fig_to_html(fig)
        plt.close(fig)
        return html
    
class PlotHelperModel(ModelPlot):
    @staticmethod
    def plot_model_pred(build_id: int, pred: pd.Series, test: pd.Series) -> str:
        my_logger.info("Starting model prediction plotting...")
        fig, ax = plt.subplots(figsize=(14, 4))
        
        ax.plot(pred.index, pred.values, label='Predicted')
        ax.plot(test.index, test.values, label='Actual')
        ax.set_title(f'Electricity consumption for (Building {build_id})')
        ax.legend(loc='upper left')

        plt.tight_layout()

        html:str = mpld3.fig_to_html(fig)
        plt.close(fig)
        return html
    
