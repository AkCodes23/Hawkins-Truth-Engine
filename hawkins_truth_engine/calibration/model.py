"""
Confidence calibration module for the Hawkins Truth Engine.

This module provides confidence calibration functionality to convert heuristic confidence scores
into calibrated probabilities using Platt scaling or isotonic regression methods.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CalibrationDataPoint(BaseModel):
    """Data point for training confidence calibration models."""
    
    features: dict[str, float] = Field(..., description="Input features (linguistic_risk, etc.)")
    heuristic_confidence: float = Field(..., ge=0.0, le=1.0, description="Original confidence score")
    true_label: bool = Field(..., description="Ground truth label")
    verdict: str = Field(..., description="Original verdict")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @validator('features')
    def validate_features(cls, v):
        """Validate that features contain numeric values."""
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric, got {type(value)}")
        return v

    @validator('verdict')
    def validate_verdict(cls, v):
        """Validate that verdict is one of the expected values."""
        valid_verdicts = {"Likely Real", "Suspicious", "Likely Fake"}
        if v not in valid_verdicts:
            raise ValueError(f"Verdict must be one of {valid_verdicts}, got '{v}'")
        return v


class CalibrationMetrics(BaseModel):
    """Metrics for evaluating calibration quality."""
    
    brier_score: float = Field(..., description="Brier score (lower is better)")
    log_loss: float = Field(..., description="Log loss (lower is better)")
    reliability_score: float = Field(..., description="Reliability diagram score")
    n_samples: int = Field(..., description="Number of samples used for evaluation")
    method: str = Field(..., description="Calibration method used")
    
    
class ConfidenceCalibrator:
    """Handles confidence calibration using Platt scaling or isotonic regression."""
    
    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        """
        Initialize the confidence calibrator.
        
        Args:
            method: Calibration method to use ("platt" or "isotonic")
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self._training_metrics = None
        self._feature_names = None
        
    def fit(self, calibration_data: list[CalibrationDataPoint]) -> None:
        """
        Train calibration model on labeled data.
        
        Args:
            calibration_data: List of calibration data points with ground truth labels
            
        Raises:
            ValueError: If calibration data is empty or invalid
        """
        if not calibration_data:
            raise ValueError("Calibration data cannot be empty")
            
        logger.info(f"Training calibration model with {len(calibration_data)} data points using {self.method} method")
        
        # Extract features and labels
        X = np.array([point.heuristic_confidence for point in calibration_data]).reshape(-1, 1)
        y = np.array([point.true_label for point in calibration_data])
        
        # Validate data
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration data must contain both positive and negative examples")
            
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create base classifier (dummy classifier that just returns the input confidence)
        base_classifier = DummyClassifier(strategy="prior")
        
        if self.method == "platt":
            # Use Platt scaling (sigmoid calibration)
            self.calibrator = CalibratedClassifierCV(
                base_classifier, 
                method="sigmoid", 
                cv="prefit"
            )
            # Fit base classifier first
            base_classifier.fit(X_train, y_train)
            # Then fit calibrator
            self.calibrator.fit(X_train, y_train)
            
        elif self.method == "isotonic":
            # Use isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(X_train.ravel(), y_train)
            
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
            
        self.is_fitted = True
        
        # Calculate training metrics
        val_predictions = self.predict_proba_batch(X_val.ravel())
        self._training_metrics = self._calculate_metrics(y_val, val_predictions)
        
        logger.info(f"Calibration model trained successfully. Validation Brier score: {self._training_metrics.brier_score:.4f}")
        
    def predict_proba(self, heuristic_confidence: float) -> float:
        """
        Convert heuristic confidence to calibrated probability.
        
        Args:
            heuristic_confidence: Original confidence score (0.0 to 1.0)
            
        Returns:
            Calibrated probability (0.0 to 1.0)
            
        Raises:
            ValueError: If model is not fitted or input is invalid
        """
        if not self.is_fitted:
            logger.warning("Calibration model not fitted, returning heuristic confidence")
            return heuristic_confidence
            
        if not 0.0 <= heuristic_confidence <= 1.0:
            raise ValueError(f"Heuristic confidence must be between 0.0 and 1.0, got {heuristic_confidence}")
            
        try:
            if self.method == "platt":
                # For Platt scaling, we need to use predict_proba
                X = np.array([[heuristic_confidence]])
                calibrated_proba = self.calibrator.predict_proba(X)[0, 1]  # Get probability of positive class
            elif self.method == "isotonic":
                # For isotonic regression, use predict directly
                calibrated_proba = self.calibrator.predict([heuristic_confidence])[0]
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
                
            # Ensure output is within valid range
            calibrated_proba = np.clip(calibrated_proba, 0.0, 1.0)
            
            return float(calibrated_proba)
            
        except Exception as e:
            logger.error(f"Error during calibration prediction: {e}")
            logger.warning("Falling back to heuristic confidence")
            return heuristic_confidence
            
    def predict_proba_batch(self, heuristic_confidences: list[float] | np.ndarray) -> np.ndarray:
        """
        Convert batch of heuristic confidences to calibrated probabilities.
        
        Args:
            heuristic_confidences: Array of original confidence scores
            
        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibration model not fitted, returning heuristic confidences")
            return np.array(heuristic_confidences)
            
        try:
            heuristic_confidences = np.array(heuristic_confidences)
            
            if self.method == "platt":
                X = heuristic_confidences.reshape(-1, 1)
                calibrated_probas = self.calibrator.predict_proba(X)[:, 1]  # Get probabilities of positive class
            elif self.method == "isotonic":
                calibrated_probas = self.calibrator.predict(heuristic_confidences)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
                
            # Ensure outputs are within valid range
            calibrated_probas = np.clip(calibrated_probas, 0.0, 1.0)
            
            return calibrated_probas
            
        except Exception as e:
            logger.error(f"Error during batch calibration prediction: {e}")
            logger.warning("Falling back to heuristic confidences")
            return np.array(heuristic_confidences)
    
    def evaluate(self, test_data: list[CalibrationDataPoint]) -> CalibrationMetrics:
        """
        Evaluate calibration quality using reliability metrics.
        
        Args:
            test_data: List of test data points with ground truth labels
            
        Returns:
            CalibrationMetrics object with evaluation results
            
        Raises:
            ValueError: If model is not fitted or test data is empty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        if not test_data:
            raise ValueError("Test data cannot be empty")
            
        # Extract features and labels
        heuristic_confidences = [point.heuristic_confidence for point in test_data]
        y_true = np.array([point.true_label for point in test_data])
        
        # Get calibrated predictions
        y_pred_proba = self.predict_proba_batch(heuristic_confidences)
        
        return self._calculate_metrics(y_true, y_pred_proba)
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> CalibrationMetrics:
        """Calculate calibration metrics."""
        try:
            # Calculate Brier score
            brier_score = brier_score_loss(y_true, y_pred_proba)
            
            # Calculate log loss
            # Add small epsilon to avoid log(0)
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            log_loss_score = log_loss(y_true, y_pred_proba_clipped)
            
            # Calculate reliability score (simplified version)
            # This is a basic implementation - could be enhanced with proper reliability diagrams
            reliability_score = self._calculate_reliability_score(y_true, y_pred_proba)
            
            return CalibrationMetrics(
                brier_score=float(brier_score),
                log_loss=float(log_loss_score),
                reliability_score=float(reliability_score),
                n_samples=len(y_true),
                method=self.method
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics if calculation fails
            return CalibrationMetrics(
                brier_score=1.0,
                log_loss=1.0,
                reliability_score=0.0,
                n_samples=len(y_true),
                method=self.method
            )
    
    def _calculate_reliability_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate reliability score based on calibration curve.
        
        This measures how well the predicted probabilities match the actual frequencies.
        A score of 1.0 indicates perfect calibration, 0.0 indicates poor calibration.
        """
        try:
            # Create bins for predicted probabilities
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            reliability_errors = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Calculate accuracy in this bin
                    accuracy_in_bin = y_true[in_bin].mean()
                    # Calculate average confidence in this bin
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    # Calculate reliability error for this bin
                    reliability_error = abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    reliability_errors.append(reliability_error)
            
            # Overall reliability error (lower is better)
            overall_reliability_error = sum(reliability_errors)
            
            # Convert to reliability score (higher is better)
            reliability_score = max(0.0, 1.0 - overall_reliability_error)
            
            return reliability_score
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.0
    
    def save_model(self, filepath: str | Path) -> None:
        """
        Save the trained calibration model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'training_metrics': self._training_metrics.dict() if self._training_metrics else None,
            'feature_names': self._feature_names,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            joblib.dump(model_data, filepath)
            logger.info(f"Calibration model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str | Path) -> None:
        """
        Load a trained calibration model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            model_data = joblib.load(filepath)
            
            self.method = model_data['method']
            self.calibrator = model_data['calibrator']
            self.is_fitted = model_data['is_fitted']
            self._feature_names = model_data.get('feature_names')
            
            # Load training metrics if available
            if model_data.get('training_metrics'):
                self._training_metrics = CalibrationMetrics(**model_data['training_metrics'])
            
            logger.info(f"Calibration model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Corrupted model file: {e}")
    
    @property
    def training_metrics(self) -> CalibrationMetrics | None:
        """Get training metrics if available."""
        return self._training_metrics
    
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'training_metrics': self._training_metrics.dict() if self._training_metrics else None,
            'feature_names': self._feature_names
        }


def load_calibration_data_from_json(filepath: str | Path) -> list[CalibrationDataPoint]:
    """
    Load calibration data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of CalibrationDataPoint objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Calibration data file not found: {filepath}")
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of calibration data points")
            
        calibration_data = []
        for i, item in enumerate(data):
            try:
                calibration_data.append(CalibrationDataPoint(**item))
            except Exception as e:
                logger.warning(f"Skipping invalid data point at index {i}: {e}")
                
        logger.info(f"Loaded {len(calibration_data)} calibration data points from {filepath}")
        return calibration_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading calibration data: {e}")


def save_calibration_data_to_json(data: list[CalibrationDataPoint], filepath: str | Path) -> None:
    """
    Save calibration data to a JSON file.
    
    Args:
        data: List of CalibrationDataPoint objects
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        json_data = [point.dict() for point in data]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(data)} calibration data points to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving calibration data: {e}")
        raise


def create_sample_calibration_data(n_samples: int = 100, random_seed: int = 42) -> list[CalibrationDataPoint]:
    """
    Create sample calibration data for testing purposes.
    
    Args:
        n_samples: Number of sample data points to create
        random_seed: Random seed for reproducibility
        
    Returns:
        List of sample CalibrationDataPoint objects
    """
    np.random.seed(random_seed)
    
    sample_data = []
    verdicts = ["Likely Real", "Suspicious", "Likely Fake"]
    
    for i in range(n_samples):
        # Generate synthetic features
        linguistic_risk = np.random.uniform(0.0, 1.0)
        statistical_risk = np.random.uniform(0.0, 1.0)
        source_trust = np.random.uniform(0.0, 1.0)
        
        # Generate heuristic confidence (somewhat correlated with features)
        heuristic_confidence = np.random.beta(2, 2)  # Beta distribution for realistic confidence scores
        
        # Generate true label (somewhat correlated with confidence)
        true_label_prob = 0.3 + 0.4 * heuristic_confidence  # Higher confidence -> more likely to be true
        true_label = np.random.random() < true_label_prob
        
        # Select verdict based on confidence
        if heuristic_confidence > 0.7:
            verdict = "Likely Real"
        elif heuristic_confidence > 0.4:
            verdict = "Suspicious"
        else:
            verdict = "Likely Fake"
        
        sample_data.append(CalibrationDataPoint(
            features={
                "linguistic_risk": linguistic_risk,
                "statistical_risk": statistical_risk,
                "source_trust": source_trust
            },
            heuristic_confidence=heuristic_confidence,
            true_label=true_label,
            verdict=verdict,
            metadata={
                "sample_id": i,
                "generated_at": datetime.now().isoformat()
            }
        ))
    
    return sample_data