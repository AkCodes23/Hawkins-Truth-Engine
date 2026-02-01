"""
Calibration module for the Hawkins Truth Engine.

This module provides confidence calibration functionality to convert heuristic confidence
scores into calibrated probabilities using machine learning techniques like Platt scaling
and isotonic regression.
"""

from .model import ConfidenceCalibrator, CalibrationDataPoint

__all__ = [
    "ConfidenceCalibrator",
    "CalibrationDataPoint",
]