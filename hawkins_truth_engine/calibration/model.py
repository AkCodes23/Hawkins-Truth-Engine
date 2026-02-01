"""
Confidence calibration module for the Hawkins Truth Engine.

This module provides confidence calibration functionality to convert heuristic confidence scores
into calibrated probabilities using Platt scaling or isotonic regression methods.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CalibrationDataPoint(BaseModel):
    """Data point for training confidence calibration models."""

    def __init__(
        self,
        features: dict[str, float] = Field(
            default_factory=dict,
            description="Input features for calibration (e.g., linguistic_risk, source_trust)",
        ),
        heuristic_confidence: float = Field(
            ..., ge=0.0, le=1.0, description="Original confidence score"
        ),
        true_label: bool = Field(..., description="Ground truth label"),
        verdict: str = Field(
            ...,
            description="Original verdict (e.g., 'Likely Real', 'Suspicious', 'Likely Fake')",
        ),
        metadata: dict[str, Any] = Field(
            default_factory=dict, description="Additional context"
        ),
    ):
        """
        Initialize the confidence calibrator.

        Args:
            method: Calibration method to use ("platt" or "isotonic")
            model_dir: Directory to store model versions (optional)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self._training_metrics = None
        self._feature_names = None
        self._model_dir = Path(model_dir) if model_dir else None
        self._current_version = None
        self._versions: dict[str, ModelVersion] = {}

        # Create model directory if specified
        if self._model_dir:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            self._load_version_history()

    def fit(
        self,
        calibration_data: list[CalibrationDataPoint],
        data_split: DataSplit | None = None,
        version: str | None = None,
        description: str = "",
    ) -> None:
        """
        Train calibration model on labeled data with enhanced data management.

        Args:
            calibration_data: List of calibration data points with ground truth labels
            data_split: Configuration for train/validation splitting
            version: Version identifier for this model
            description: Description of this model version

        Raises:
            ValueError: If calibration data is empty or invalid
        """
        if not calibration_data:
            raise ValueError("Calibration data cannot be empty")

        # Validate data
        validation_result = self.validate_data(calibration_data)
        if not validation_result.is_valid:
            error_msg = f"Data validation failed: {'; '.join(validation_result.errors)}"
            raise ValueError(error_msg)

        # Log validation warnings
        for warning in validation_result.warnings:
            logger.warning(f"Data validation warning: {warning}")

        logger.info(
            f"Training calibration model with {len(calibration_data)} data points using {self.method} method"
        )

        # Use default split if not provided
        if data_split is None:
            data_split = DataSplit(train_size=0.8, validation_size=0.2)

        # Split data
        train_data, val_data = self._split_data(calibration_data, data_split)

        # Extract features and labels
        X_train = np.array(
            [point.heuristic_confidence for point in train_data]
        ).reshape(-1, 1)
        y_train = np.array([point.true_label for point in train_data])
        X_val = np.array([point.heuristic_confidence for point in val_data]).reshape(
            -1, 1
        )
        y_val = np.array([point.true_label for point in val_data])

        # Create base classifier
        base_classifier = LogisticRegression()

        if self.method == "platt":
            # Use Platt scaling (sigmoid calibration)
            self.calibrator = CalibratedClassifierCV(
                base_classifier,
                method="sigmoid",
                cv=3,  # Use 3-fold cross-validation
            )
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

        # Save version if model directory is configured
        if self._model_dir and version:
            self._save_version(calibration_data, data_split, version, description)

        logger.info(
            f"Calibration model trained successfully. Validation Brier score: {self._training_metrics.brier_score:.4f}"
        )

    def validate_data(self, data: list[CalibrationDataPoint]) -> DataValidationResult:
        """
        Validate calibration data for completeness and consistency.

        Args:
            data: List of calibration data points to validate

        Returns:
            DataValidationResult with validation details
        """
        if not data:
            return DataValidationResult(
                is_valid=False,
                total_samples=0,
                positive_samples=0,
                negative_samples=0,
                errors=["Data is empty"],
            )

        total_samples = len(data)
        positive_samples = sum(1 for point in data if point.true_label)
        negative_samples = total_samples - positive_samples

        errors = []
        warnings = []
        invalid_samples = []
        missing_features = set()

        # Check for class balance
        if positive_samples == 0:
            errors.append("No positive examples found")
        elif negative_samples == 0:
            errors.append("No negative examples found")
        elif min(positive_samples, negative_samples) / total_samples < 0.05:
            warnings.append(
                f"Severe class imbalance: {positive_samples} positive, {negative_samples} negative"
            )

        # Check individual data points
        all_feature_keys = set()
        for i, point in enumerate(data):
            try:
                # Validate confidence range
                if not (0.0 <= point.heuristic_confidence <= 1.0):
                    invalid_samples.append(i)
                    errors.append(
                        f"Sample {i}: Invalid confidence {point.heuristic_confidence}"
                    )

                # Collect feature keys
                all_feature_keys.update(point.features.keys())

                # Check for missing or invalid features
                for key, value in point.features.items():
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        missing_features.add(key)

            except Exception as e:
                invalid_samples.append(i)
                errors.append(f"Sample {i}: Validation error - {e}")

        # Check feature consistency
        for i, point in enumerate(data):
            missing_in_sample = all_feature_keys - set(point.features.keys())
            if missing_in_sample:
                warnings.append(f"Sample {i}: Missing features {missing_in_sample}")

        # Check minimum sample size
        if total_samples < 20:
            warnings.append(
                f"Small dataset size: {total_samples} samples (recommended: >100)"
            )
        elif total_samples < 100:
            warnings.append(
                f"Moderate dataset size: {total_samples} samples (recommended: >100)"
            )

        is_valid = len(errors) == 0

        return DataValidationResult(
            is_valid=is_valid,
            total_samples=total_samples,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            missing_features=list(missing_features),
            invalid_samples=invalid_samples,
            warnings=warnings,
            errors=errors,
        )

    def _split_data(
        self, data: list[CalibrationDataPoint], split_config: DataSplit
    ) -> tuple[list[CalibrationDataPoint], list[CalibrationDataPoint]]:
        """Split data into train and validation sets."""
        # Check if we have enough samples for stratification
        labels = [point.true_label for point in data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_class_count = min(counts)

        # Disable stratification if any class has fewer than 2 samples
        use_stratify = split_config.stratify and min_class_count >= 2

        if not use_stratify and split_config.stratify:
            logger.warning(
                f"Disabling stratification due to insufficient samples in some classes (min: {min_class_count})"
            )

        if split_config.test_size > 0:
            # Three-way split
            train_val_size = split_config.train_size + split_config.validation_size
            train_ratio = split_config.train_size / train_val_size

            # First split: separate test set
            if use_stratify:
                train_val_data, test_data = train_test_split(
                    data,
                    test_size=split_config.test_size,
                    random_state=split_config.random_seed,
                    stratify=labels,
                )
                # Second split: separate train and validation
                train_val_labels = [point.true_label for point in train_val_data]
                train_data, val_data = train_test_split(
                    train_val_data,
                    train_size=train_ratio,
                    random_state=split_config.random_seed,
                    stratify=train_val_labels,
                )
            else:
                train_val_data, test_data = train_test_split(
                    data,
                    test_size=split_config.test_size,
                    random_state=split_config.random_seed,
                )
                train_data, val_data = train_test_split(
                    train_val_data,
                    train_size=train_ratio,
                    random_state=split_config.random_seed,
                )
        else:
            # Two-way split
            if use_stratify:
                train_data, val_data = train_test_split(
                    data,
                    train_size=split_config.train_size,
                    random_state=split_config.random_seed,
                    stratify=labels,
                )
            else:
                train_data, val_data = train_test_split(
                    data,
                    train_size=split_config.train_size,
                    random_state=split_config.random_seed,
                )

        return train_data, val_data

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
            logger.warning(
                "Calibration model not fitted, returning heuristic confidence"
            )
            return heuristic_confidence

        if not 0.0 <= heuristic_confidence <= 1.0:
            raise ValueError(
                f"Heuristic confidence must be between 0.0 and 1.0, got {heuristic_confidence}"
            )

        try:
            if self.method == "platt":
                # For Platt scaling, we need to use predict_proba
                X = np.array([[heuristic_confidence]])
                calibrated_proba = self.calibrator.predict_proba(X)[
                    0, 1
                ]  # Get probability of positive class
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

    def predict_proba_batch(
        self, heuristic_confidences: list[float] | np.ndarray
    ) -> np.ndarray:
        """
        Convert batch of heuristic confidences to calibrated probabilities.

        Args:
            heuristic_confidences: Array of original confidence scores

        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning(
                "Calibration model not fitted, returning heuristic confidences"
            )
            return np.array(heuristic_confidences)

        try:
            heuristic_confidences = np.array(heuristic_confidences)

            if self.method == "platt":
                X = heuristic_confidences.reshape(-1, 1)
                calibrated_probas = self.calibrator.predict_proba(X)[
                    :, 1
                ]  # Get probabilities of positive class
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

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> CalibrationMetrics:
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
                method=self.method,
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics if calculation fails
            return CalibrationMetrics(
                brier_score=1.0,
                log_loss=1.0,
                reliability_score=0.0,
                n_samples=len(y_true),
                method=self.method,
            )

    def _calculate_reliability_score(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> float:
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
                    reliability_error = (
                        abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    )
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
            "method": self.method,
            "calibrator": self.calibrator,
            "is_fitted": self.is_fitted,
            "training_metrics": self._training_metrics.model_dump()
            if self._training_metrics
            else None,
            "feature_names": self._feature_names,
            "saved_at": datetime.now().isoformat(),
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

            self.method = model_data["method"]
            self.calibrator = model_data["calibrator"]
            self.is_fitted = model_data["is_fitted"]
            self._feature_names = model_data.get("feature_names")

            # Load training metrics if available
            if model_data.get("training_metrics"):
                self._training_metrics = CalibrationMetrics(
                    **model_data["training_metrics"]
                )

            logger.info(f"Calibration model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Corrupted model file: {e}")

    def _save_version(
        self,
        data: list[CalibrationDataPoint],
        split_config: DataSplit,
        version: str,
        description: str,
    ) -> None:
        """Save a model version with metadata."""
        if not self._model_dir:
            return

        # Calculate data hash for reproducibility
        data_hash = self._calculate_data_hash(data)

        # Create version directory
        version_dir = self._model_dir / version
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / "model.joblib"
        self.save_model(model_path)

        # Save configuration
        config = {
            "method": self.method,
            "split_config": split_config.model_dump(),
            "data_size": len(data),
        }

        # Create version metadata
        model_version = ModelVersion(
            version=version,
            description=description,
            metrics=self._training_metrics,
            data_hash=data_hash,
            model_path=str(model_path.relative_to(self._model_dir)),
            config=config,
        )

        # Save version metadata
        version_file = version_dir / "version.json"
        with open(version_file, "w") as f:
            json.dump(model_version.model_dump(), f, indent=2, default=str)

        # Update version history
        self._versions[version] = model_version
        self._current_version = version
        self._save_version_history()

        logger.info(f"Model version {version} saved to {version_dir}")

    def _calculate_data_hash(self, data: list[CalibrationDataPoint]) -> str:
        """Calculate hash of training data for reproducibility tracking."""
        import hashlib

        # Create a deterministic string representation of the data
        data_str = ""
        for point in sorted(data, key=lambda x: (x.heuristic_confidence, x.true_label)):
            data_str += (
                f"{point.heuristic_confidence}:{point.true_label}:{point.verdict};"
            )

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _load_version_history(self) -> None:
        """Load version history from disk."""
        if not self._model_dir:
            return

        history_file = self._model_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history_data = json.load(f)

                for version_data in history_data.get("versions", []):
                    version = ModelVersion(**version_data)
                    self._versions[version.version] = version

                self._current_version = history_data.get("current_version")

            except Exception as e:
                logger.warning(f"Error loading version history: {e}")

    def _save_version_history(self) -> None:
        """Save version history to disk."""
        if not self._model_dir:
            return

        history_data = {
            "current_version": self._current_version,
            "versions": [v.model_dump() for v in self._versions.values()],
        }

        history_file = self._model_dir / "version_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving version history: {e}")

    def list_versions(self) -> list[ModelVersion]:
        """List all available model versions."""
        return list(self._versions.values())

    def rollback_to_version(self, version: str) -> None:
        """
        Rollback to a specific model version.

        Args:
            version: Version identifier to rollback to

        Raises:
            ValueError: If version doesn't exist
        """
        if version not in self._versions:
            available = list(self._versions.keys())
            raise ValueError(
                f"Version {version} not found. Available versions: {available}"
            )

        version_info = self._versions[version]
        model_path = self._model_dir / version_info.model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for version {version}: {model_path}"
            )

        # Load the model
        self.load_model(model_path)
        self._current_version = version

        logger.info(f"Rolled back to model version {version}")

    def get_current_version(self) -> str | None:
        """Get the current model version."""
        return self._current_version

    def compare_versions(self, version1: str, version2: str) -> dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with comparison results
        """
        if version1 not in self._versions or version2 not in self._versions:
            raise ValueError("One or both versions not found")

        v1 = self._versions[version1]
        v2 = self._versions[version2]

        comparison = {
            "version1": {
                "version": v1.version,
                "created_at": v1.created_at,
                "metrics": v1.metrics.model_dump() if v1.metrics else None,
                "config": v1.config,
            },
            "version2": {
                "version": v2.version,
                "created_at": v2.created_at,
                "metrics": v2.metrics.model_dump() if v2.metrics else None,
                "config": v2.config,
            },
            "metrics_comparison": {},
        }

        # Compare metrics if both versions have them
        if v1.metrics and v2.metrics:
            comparison["metrics_comparison"] = {
                "brier_score_diff": v2.metrics.brier_score - v1.metrics.brier_score,
                "log_loss_diff": v2.metrics.log_loss - v1.metrics.log_loss,
                "reliability_score_diff": v2.metrics.reliability_score
                - v1.metrics.reliability_score,
                "better_version": version1
                if v1.metrics.brier_score < v2.metrics.brier_score
                else version2,
            }

        return comparison

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
            "method": self.method,
            "is_fitted": self.is_fitted,
            "training_metrics": self._training_metrics.model_dump()
            if self._training_metrics
            else None,
            "feature_names": self._feature_names,
        }


def load_calibration_data_from_csv(
    filepath: str | Path, feature_columns: list[str] | None = None
) -> list[CalibrationDataPoint]:
    """
    Load calibration data from a CSV file.

    Args:
        filepath: Path to the CSV file
        feature_columns: List of column names to use as features (if None, auto-detect)

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
        # Read CSV file
        # Use round-trip float parsing to preserve full precision from CSV.
        df = pd.read_csv(filepath, float_precision="round_trip")

        # Validate required columns
        required_columns = ["heuristic_confidence", "true_label", "verdict"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Auto-detect feature columns if not specified
        if feature_columns is None:
            # Use all numeric columns except the required ones
            excluded_columns = set(required_columns + ["metadata"])
            feature_columns = [
                col
                for col in df.columns
                if col not in excluded_columns and df[col].dtype in ["int64", "float64"]
            ]

        logger.info(f"Using feature columns: {feature_columns}")

        calibration_data = []
        for i, row in df.iterrows():
            try:
                # Extract features
                features = {}
                for col in feature_columns:
                    if col in row and pd.notna(row[col]):
                        features[col] = float(row[col])

                # Extract metadata if present
                metadata = {}
                if "metadata" in row and pd.notna(row["metadata"]):
                    try:
                        metadata = (
                            json.loads(row["metadata"])
                            if isinstance(row["metadata"], str)
                            else {}
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid metadata JSON at row {i}, skipping metadata"
                        )

                # Add row index to metadata
                metadata["csv_row"] = int(i)

                calibration_data.append(
                    CalibrationDataPoint(
                        features=features,
                        heuristic_confidence=float(row["heuristic_confidence"]),
                        true_label=bool(row["true_label"]),
                        verdict=str(row["verdict"]),
                        metadata=metadata,
                    )
                )

            except Exception as e:
                logger.warning(f"Skipping invalid data point at row {i}: {e}")

        logger.info(
            f"Loaded {len(calibration_data)} calibration data points from {filepath}"
        )
        return calibration_data

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading calibration data: {e}")


def save_calibration_data_to_csv(
    data: list[CalibrationDataPoint], filepath: str | Path
) -> None:
    """
    Save calibration data to a CSV file.

    Args:
        data: List of CalibrationDataPoint objects
        filepath: Path to save the CSV file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        raise ValueError("Cannot save empty data")

    try:
        # Collect all feature names
        all_features = set()
        for point in data:
            all_features.update(point.features.keys())
        all_features = sorted(all_features)

        # Prepare data for DataFrame
        rows = []
        for point in data:
            row = {
                "heuristic_confidence": point.heuristic_confidence,
                "true_label": point.true_label,
                "verdict": point.verdict,
                "metadata": json.dumps(point.metadata) if point.metadata else "",
            }

            # Add feature columns
            for feature in all_features:
                row[feature] = point.features.get(feature, np.nan)

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        # Reorder columns for better readability
        column_order = (
            ["heuristic_confidence", "true_label", "verdict"]
            + all_features
            + ["metadata"]
        )
        df = df[column_order]

        # Preserve float round-trips across CSV load/save.
        # Pandas' default float formatting can truncate precision and break strict equality tests.
        df.to_csv(filepath, index=False, float_format="%.17g")
        logger.info(f"Saved {len(data)} calibration data points to {filepath}")

    except Exception as e:
        logger.error(f"Error saving calibration data: {e}")
        raise


def load_calibration_data(
    filepath: str | Path, format: Literal["auto", "json", "csv"] = "auto", **kwargs
) -> list[CalibrationDataPoint]:
    """
    Load calibration data from JSON or CSV format with auto-detection.

    Args:
        filepath: Path to the data file
        format: File format ("auto", "json", or "csv")
        **kwargs: Additional arguments passed to format-specific loaders

    Returns:
        List of CalibrationDataPoint objects
    """
    filepath = Path(filepath)

    if format == "auto":
        # Auto-detect format based on file extension
        if filepath.suffix.lower() == ".json":
            format = "json"
        elif filepath.suffix.lower() == ".csv":
            format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for file: {filepath}")

    if format == "json":
        return load_calibration_data_from_json(filepath)
    elif format == "csv":
        return load_calibration_data_from_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_calibration_data(
    data: list[CalibrationDataPoint],
    filepath: str | Path,
    format: Literal["auto", "json", "csv"] = "auto",
) -> None:
    """
    Save calibration data to JSON or CSV format with auto-detection.

    Args:
        data: List of CalibrationDataPoint objects
        filepath: Path to save the data file
        format: File format ("auto", "json", or "csv")
    """
    filepath = Path(filepath)

    if format == "auto":
        # Auto-detect format based on file extension
        if filepath.suffix.lower() == ".json":
            format = "json"
        elif filepath.suffix.lower() == ".csv":
            format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for file: {filepath}")

    if format == "json":
        save_calibration_data_to_json(data, filepath)
    elif format == "csv":
        save_calibration_data_to_csv(data, filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_calibration_datasets(
    *datasets: list[CalibrationDataPoint],
) -> list[CalibrationDataPoint]:
    """
    Merge multiple calibration datasets into one.

    Args:
        *datasets: Variable number of calibration datasets

    Returns:
        Merged dataset with duplicate detection
    """
    if not datasets:
        return []

    merged = []
    seen_hashes = set()

    for dataset in datasets:
        for point in dataset:
            # Create a hash for duplicate detection
            point_hash = hash(
                (
                    point.heuristic_confidence,
                    point.true_label,
                    point.verdict,
                    tuple(sorted(point.features.items())),
                )
            )

            if point_hash not in seen_hashes:
                merged.append(point)
                seen_hashes.add(point_hash)
            else:
                logger.debug(
                    f"Skipping duplicate data point: {point.heuristic_confidence}"
                )

    logger.info(
        f"Merged {sum(len(d) for d in datasets)} data points into {len(merged)} unique points"
    )
    return merged


def split_calibration_data(
    data: list[CalibrationDataPoint], split_config: DataSplit
) -> dict[str, list[CalibrationDataPoint]]:
    """
    Split calibration data into train/validation/test sets.

    Args:
        data: List of calibration data points
        split_config: Configuration for data splitting

    Returns:
        Dictionary with 'train', 'validation', and optionally 'test' keys
    """
    if not data:
        raise ValueError("Cannot split empty data")

    # Prepare for splitting
    # Check if we have enough samples for stratification
    labels = [point.true_label for point in data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_class_count = min(counts)

    # Disable stratification if any class has fewer than 2 samples
    use_stratify = split_config.stratify and min_class_count >= 2

    if not use_stratify and split_config.stratify:
        logger.warning(
            f"Disabling stratification due to insufficient samples in some classes (min: {min_class_count})"
        )

    if split_config.test_size > 0:
        # Three-way split with deterministic sizes.
        # Using float ratios with sklearn can lead to off-by-one rounding differences.
        n = len(data)
        n_test = int(round(n * float(split_config.test_size)))
        n_val = int(round(n * float(split_config.validation_size)))
        n_test = max(0, min(n - 1, n_test))
        n_val = max(0, min(n - n_test - 1, n_val))
        n_train = n - n_test - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(
                f"Invalid split sizes from config: train={n_train}, validation={n_val}, test={n_test} (n={n})"
            )

        if use_stratify:
            train_val_data, test_data = train_test_split(
                data,
                test_size=n_test,
                random_state=split_config.random_seed,
                stratify=labels,
            )
            train_val_labels = [point.true_label for point in train_val_data]
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=n_val,
                random_state=split_config.random_seed,
                stratify=train_val_labels,
            )
        else:
            train_val_data, test_data = train_test_split(
                data,
                test_size=n_test,
                random_state=split_config.random_seed,
            )
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=n_val,
                random_state=split_config.random_seed,
            )

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }
    else:
        # Two-way split
        if use_stratify:
            train_data, val_data = train_test_split(
                data,
                train_size=split_config.train_size,
                random_state=split_config.random_seed,
                stratify=labels,
            )
        else:
            train_data, val_data = train_test_split(
                data,
                train_size=split_config.train_size,
                random_state=split_config.random_seed,
            )

        return {"train": train_data, "validation": val_data}


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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of calibration data points")

        calibration_data = []
        for i, item in enumerate(data):
            try:
                calibration_data.append(CalibrationDataPoint(**item))
            except Exception as e:
                logger.warning(f"Skipping invalid data point at index {i}: {e}")

        logger.info(
            f"Loaded {len(calibration_data)} calibration data points from {filepath}"
        )
        return calibration_data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading calibration data: {e}")


def save_calibration_data_to_json(
    data: list[CalibrationDataPoint], filepath: str | Path
) -> None:
    """
    Save calibration data to a JSON file.

    Args:
        data: List of CalibrationDataPoint objects
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_data = [point.model_dump() for point in data]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} calibration data points to {filepath}")

    except Exception as e:
        logger.error(f"Error saving calibration data: {e}")
        raise


def create_sample_calibration_data(
    n_samples: int = 100, random_seed: int = 42, include_features: bool = True
) -> list[CalibrationDataPoint]:
    """
    Create sample calibration data for testing purposes.

    Args:
        n_samples: Number of sample data points to create
        random_seed: Random seed for reproducibility
        include_features: Whether to include additional features beyond confidence

    Returns:
        List of sample CalibrationDataPoint objects
    """
    np.random.seed(random_seed)

    sample_data = []
    verdicts = ["Likely Real", "Suspicious", "Likely Fake"]

    for i in range(n_samples):
        # Generate synthetic features
        features = {}
        if include_features:
            features.update(
                {
                    "linguistic_risk": np.random.uniform(0.0, 1.0),
                    "statistical_risk": np.random.uniform(0.0, 1.0),
                    "source_trust": np.random.uniform(0.0, 1.0),
                    "external_corroboration": np.random.uniform(0.0, 1.0),
                }
            )

        # Generate heuristic confidence (somewhat correlated with features)
        heuristic_confidence = np.random.beta(
            2, 2
        )  # Beta distribution for realistic confidence scores

        # Generate true label (somewhat correlated with confidence)
        true_label_prob = (
            0.3 + 0.4 * heuristic_confidence
        )  # Higher confidence -> more likely to be true
        true_label = np.random.random() < true_label_prob

        # Select verdict based on confidence
        if heuristic_confidence > 0.7:
            verdict = "Likely Real"
        elif heuristic_confidence > 0.4:
            verdict = "Suspicious"
        else:
            verdict = "Likely Fake"

        sample_data.append(
            CalibrationDataPoint(
                features=features,
                heuristic_confidence=heuristic_confidence,
                true_label=true_label,
                verdict=verdict,
                metadata={
                    "sample_id": i,
                    "generated_at": datetime.now().isoformat(),
                    "source": "synthetic",
                },
            )
        )

    return sample_data
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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of calibration data points")

        calibration_data = []
        for i, item in enumerate(data):
            try:
                calibration_data.append(CalibrationDataPoint(**item))
            except Exception as e:
                logger.warning(f"Skipping invalid data point at index {i}: {e}")

        logger.info(
            f"Loaded {len(calibration_data)} calibration data points from {filepath}"
        )
        return calibration_data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading calibration data: {e}")


def save_calibration_data_to_json(
    data: list[CalibrationDataPoint], filepath: str | Path
) -> None:
    """
    Save calibration data to a JSON file.

    Args:
        data: List of CalibrationDataPoint objects
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_data = [point.model_dump() for point in data]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} calibration data points to {filepath}")

    except Exception as e:
        logger.error(f"Error saving calibration data: {e}")
        raise


def create_sample_calibration_data(
    n_samples: int = 100, random_seed: int = 42
) -> list[CalibrationDataPoint]:
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
        heuristic_confidence = np.random.beta(
            2, 2
        )  # Beta distribution for realistic confidence scores

        # Generate true label (somewhat correlated with confidence)
        true_label_prob = (
            0.3 + 0.4 * heuristic_confidence
        )  # Higher confidence -> more likely to be true
        true_label = np.random.random() < true_label_prob

        # Select verdict based on confidence
        if heuristic_confidence > 0.7:
            verdict = "Likely Real"
        elif heuristic_confidence > 0.4:
            verdict = "Suspicious"
        else:
            verdict = "Likely Fake"

        sample_data.append(
            CalibrationDataPoint(
                features={
                    "linguistic_risk": linguistic_risk,
                    "statistical_risk": statistical_risk,
                    "source_trust": source_trust,
                },
                heuristic_confidence=heuristic_confidence,
                true_label=true_label,
                verdict=verdict,
                metadata={"sample_id": i, "generated_at": datetime.now().isoformat()},
            )
        )

    return sample_data
