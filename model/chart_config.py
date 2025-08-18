from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from model.context import UnitOfAnalysis


@dataclass
class ChartConfig:
    title: Optional[str] = None
    y_label: Optional[str] = None
    x_label: Optional[str] = None


class ChartConfigBuilder:
    def __init__(self):
        self.chartConfig = ChartConfig()
        self.chartConfig.title = "Chart title"
        self.chartConfig.x_label = "X label"
        self.chartConfig.y_label = "Y label"

    def with_title(self, title: str):
        self.chartConfig.title = title
        return self  # Return self to allow method chaining

    def with_x_label(self, label: str):
        self.chartConfig.x_label = label
        return self

    def with_y_label(self, label: str):
        self.chartConfig.y_label = label
        return self

    def build(self) -> ChartConfig:
        """
        Constructs and returns the chartConfig object.
        """
        # Instantiate the chartConfig dataclass and return it
        return self.chartConfig
