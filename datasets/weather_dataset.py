"""Public import path for the unified manifest-backed weather dataset."""
from airformer.data import PairedWeatherDataset

WeatherDataset = PairedWeatherDataset
__all__ = ["WeatherDataset", "PairedWeatherDataset"]
