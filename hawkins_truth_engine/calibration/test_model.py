"""
Unit tests for the calibration model module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hawkins_truth_engine.calibration.model import (
    CalibrationDataPoint,
    CalibrationMetrics,
    ConfidenceCalibrator,
    DataSplit,
    DataValidationResult,
    ModelVersion,
    create_sample_calibration_data,
    load_calibration_data,
    load_calibration_data_from_csv,
    load_calibration_data_from_json,
    merge_calibration_datasets,
    save_calibration_data,
    save_calibration_data_to_csv,
    save_calibration_data_to_json,
    split_calibration_data,
)


class TestCalibrationDataPoint:
    """Test CalibrationDataPoint schema validation."""
    
    def test_valid_data_point(self):
        """Test creating a valid calibration data point."""
        data_point = CalibrationDataPoint(
            features={"linguistic_risk": 0.5, "source_trust": 0.8},
            heuristic_confidence=0.7,
            true_label=True,
            verdict="Likely Real",
            metadata={"test": "value"}
        )
        
        assert data_point.features["linguistic_risk"] == 0.5
        assert data_point.heuristic_confidence == 0.7
        assert data_point.true_label is True
        assert data_point.verdict == "Likely Real"
    
    def test_invalid_confidence_range(self):
        """Test validation of confidence range."""
        with pytest.raises(ValueError):
            CalibrationDataPoint(
                features={"linguistic_risk": 0.5},
                heuristic_confidence=1.5,  # Invalid: > 1.0
                true_label=True,
                verdict="Likely Real"
            )
    
    def test_invalid_verdict(self):
        """Test validation of verdict values."""
        with pytest.raises(ValueError):
            CalibrationDataPoint(
                features={"linguistic_risk": 0.5},
                heuristic_confidence=0.7,
                true_label=True,
                verdict="Invalid Verdict"  # Invalid verdict
            )
    
    def test_invalid_feature_type(self):
        """Test validation of feature types."""
        with pytest.raises(ValueError):
            CalibrationDataPoint(
                features={"linguistic_risk": "not_a_number"},  # Invalid: string instead of float
                heuristic_confidence=0.7,
                true_label=True,
                verdict="Likely Real"
            )


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator functionality."""
    
    def test_initialization(self):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator(method="platt")
        assert calibrator.method == "platt"
        assert not calibrator.is_fitted
        
        calibrator_isotonic = ConfidenceCalibrator(method="isotonic")
        assert calibrator_isotonic.method == "isotonic"
    
    def test_invalid_method(self):
        """Test initialization with invalid method."""
        calibrator = ConfidenceCalibrator(method="invalid")
        sample_data = create_sample_calibration_data(n_samples=50)
        
        with pytest.raises(ValueError):
            calibrator.fit(sample_data)
    
    def test_fit_platt_scaling(self):
        """Test fitting with Platt scaling."""
        calibrator = ConfidenceCalibrator(method="platt")
        sample_data = create_sample_calibration_data(n_samples=100)
        
        calibrator.fit(sample_data)
        
        assert calibrator.is_fitted
        assert calibrator.calibrator is not None
        assert calibrator.training_metrics is not None
    
    def test_fit_isotonic_regression(self):
        """Test fitting with isotonic regression."""
        calibrator = ConfidenceCalibrator(method="isotonic")
        sample_data = create_sample_calibration_data(n_samples=100)
        
        calibrator.fit(sample_data)
        
        assert calibrator.is_fitted
        assert calibrator.calibrator is not None
        assert calibrator.training_metrics is not None
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        calibrator = ConfidenceCalibrator()
        
        with pytest.raises(ValueError):
            calibrator.fit([])
    
    def test_fit_single_class_data(self):
        """Test fitting with data containing only one class."""
        calibrator = ConfidenceCalibrator()
        
        # Create data with only positive examples
        single_class_data = [
            CalibrationDataPoint(
                features={"risk": 0.5},
                heuristic_confidence=0.7,
                true_label=True,  # All True
                verdict="Likely Real"
            )
            for _ in range(10)
        ]
        
        with pytest.raises(ValueError):
            calibrator.fit(single_class_data)
    
    def test_predict_proba_unfitted(self):
        """Test prediction with unfitted model."""
        calibrator = ConfidenceCalibrator()
        
        # Should return heuristic confidence when not fitted
        result = calibrator.predict_proba(0.7)
        assert result == 0.7
    
    def test_predict_proba_fitted(self):
        """Test prediction with fitted model."""
        calibrator = ConfidenceCalibrator(method="platt")
        sample_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(sample_data)
        
        result = calibrator.predict_proba(0.7)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_predict_proba_invalid_input(self):
        """Test prediction with invalid input."""
        calibrator = ConfidenceCalibrator()
        sample_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(sample_data)
        
        with pytest.raises(ValueError):
            calibrator.predict_proba(1.5)  # Invalid: > 1.0
        
        with pytest.raises(ValueError):
            calibrator.predict_proba(-0.1)  # Invalid: < 0.0
    
    def test_predict_proba_batch(self):
        """Test batch prediction."""
        calibrator = ConfidenceCalibrator(method="isotonic")
        sample_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(sample_data)
        
        confidences = [0.3, 0.5, 0.7, 0.9]
        results = calibrator.predict_proba_batch(confidences)
        
        assert len(results) == len(confidences)
        assert all(0.0 <= r <= 1.0 for r in results)
    
    def test_evaluate(self):
        """Test model evaluation."""
        calibrator = ConfidenceCalibrator(method="platt")
        train_data = create_sample_calibration_data(n_samples=100, random_seed=42)
        test_data = create_sample_calibration_data(n_samples=50, random_seed=123)
        
        calibrator.fit(train_data)
        metrics = calibrator.evaluate(test_data)
        
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.n_samples == 50
        assert metrics.method == "platt"
        assert 0.0 <= metrics.brier_score <= 1.0
        assert 0.0 <= metrics.reliability_score <= 1.0
    
    def test_evaluate_unfitted(self):
        """Test evaluation with unfitted model."""
        calibrator = ConfidenceCalibrator()
        test_data = create_sample_calibration_data(n_samples=50)
        
        with pytest.raises(ValueError):
            calibrator.evaluate(test_data)
    
    def test_evaluate_empty_data(self):
        """Test evaluation with empty test data."""
        calibrator = ConfidenceCalibrator()
        train_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(train_data)
        
        with pytest.raises(ValueError):
            calibrator.evaluate([])
    
    def test_save_load_model(self):
        """Test model persistence."""
        calibrator = ConfidenceCalibrator(method="platt")
        sample_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(sample_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            # Save model
            calibrator.save_model(model_path)
            assert model_path.exists()
            
            # Load model into new calibrator
            new_calibrator = ConfidenceCalibrator()
            new_calibrator.load_model(model_path)
            
            assert new_calibrator.is_fitted
            assert new_calibrator.method == "platt"
            
            # Test that predictions are the same
            test_confidence = 0.7
            original_pred = calibrator.predict_proba(test_confidence)
            loaded_pred = new_calibrator.predict_proba(test_confidence)
            
            assert abs(original_pred - loaded_pred) < 1e-6
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model."""
        calibrator = ConfidenceCalibrator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            with pytest.raises(ValueError):
                calibrator.save_model(model_path)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model."""
        calibrator = ConfidenceCalibrator()
        
        with pytest.raises(FileNotFoundError):
            calibrator.load_model("nonexistent_model.joblib")
    
    def test_get_model_info(self):
        """Test getting model information."""
        calibrator = ConfidenceCalibrator(method="isotonic")
        
        # Before fitting
        info = calibrator.get_model_info()
        assert info["method"] == "isotonic"
        assert not info["is_fitted"]
        assert info["training_metrics"] is None
        
        # After fitting
        sample_data = create_sample_calibration_data(n_samples=100)
        calibrator.fit(sample_data)
        
        info = calibrator.get_model_info()
        assert info["is_fitted"]
        assert info["training_metrics"] is not None


class TestDataManagement:
    """Test calibration data management functions."""
    
    def test_create_sample_data(self):
        """Test creating sample calibration data."""
        sample_data = create_sample_calibration_data(n_samples=50, random_seed=42)
        
        assert len(sample_data) == 50
        assert all(isinstance(point, CalibrationDataPoint) for point in sample_data)
        
        # Test reproducibility
        sample_data2 = create_sample_calibration_data(n_samples=50, random_seed=42)
        assert len(sample_data2) == 50
        
        # Should be the same due to same random seed
        for p1, p2 in zip(sample_data, sample_data2):
            assert p1.heuristic_confidence == p2.heuristic_confidence
            assert p1.true_label == p2.true_label
    
    def test_save_load_json(self):
        """Test saving and loading calibration data to/from JSON."""
        sample_data = create_sample_calibration_data(n_samples=20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test_data.json"
            
            # Save data
            save_calibration_data_to_json(sample_data, json_path)
            assert json_path.exists()
            
            # Load data
            loaded_data = load_calibration_data_from_json(json_path)
            
            assert len(loaded_data) == len(sample_data)
            
            # Compare first data point
            original = sample_data[0]
            loaded = loaded_data[0]
            
            assert loaded.heuristic_confidence == original.heuristic_confidence
            assert loaded.true_label == original.true_label
            assert loaded.verdict == original.verdict
    
    def test_load_nonexistent_json(self):
        """Test loading non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_calibration_data_from_json("nonexistent.json")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "invalid.json"
            
            # Create invalid JSON
            with open(json_path, 'w') as f:
                f.write("invalid json content")
            
            with pytest.raises(ValueError):
                load_calibration_data_from_json(json_path)
    
    def test_load_invalid_format_json(self):
        """Test loading JSON with invalid format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "invalid_format.json"
            
            # Create JSON that's not a list
            with open(json_path, 'w') as f:
                json.dump({"not": "a list"}, f)
            
            with pytest.raises(ValueError):
                load_calibration_data_from_json(json_path)


class TestDataSplit:
    """Test DataSplit configuration validation."""
    
    def test_valid_split(self):
        """Test creating a valid data split configuration."""
        split = DataSplit(train_size=0.7, validation_size=0.3)
        assert split.train_size == 0.7
        assert split.validation_size == 0.3
        assert split.test_size == 0.0
        assert split.stratify is True
        
    def test_three_way_split(self):
        """Test three-way split configuration."""
        split = DataSplit(train_size=0.6, validation_size=0.2, test_size=0.2)
        assert split.train_size == 0.6
        assert split.validation_size == 0.2
        assert split.test_size == 0.2
        
    def test_invalid_split_sum(self):
        """Test validation of split sizes that don't sum to 1.0."""
        with pytest.raises(ValueError):
            DataSplit(train_size=0.6, validation_size=0.6)  # Sum = 1.2
            
    def test_split_size_bounds(self):
        """Test validation of split size bounds."""
        with pytest.raises(ValueError):
            DataSplit(train_size=0.05, validation_size=0.95)  # train_size too small
            
        with pytest.raises(ValueError):
            DataSplit(train_size=0.95, validation_size=0.05)  # validation_size too small


class TestDataValidation:
    """Test data validation functionality."""
    
    def test_valid_data_validation(self):
        """Test validation of valid data."""
        calibrator = ConfidenceCalibrator()
        sample_data = create_sample_calibration_data(n_samples=100)
        
        result = calibrator.validate_data(sample_data)
        
        assert result.is_valid
        assert result.total_samples == 100
        assert result.positive_samples > 0
        assert result.negative_samples > 0
        assert len(result.errors) == 0
        
    def test_empty_data_validation(self):
        """Test validation of empty data."""
        calibrator = ConfidenceCalibrator()
        
        result = calibrator.validate_data([])
        
        assert not result.is_valid
        assert result.total_samples == 0
        assert "Data is empty" in result.errors
        
    def test_single_class_validation(self):
        """Test validation of data with only one class."""
        calibrator = ConfidenceCalibrator()
        
        # Create data with only positive examples
        single_class_data = [
            CalibrationDataPoint(
                features={"risk": 0.5},
                heuristic_confidence=0.7,
                true_label=True,  # All True
                verdict="Likely Real"
            )
            for _ in range(10)
        ]
        
        result = calibrator.validate_data(single_class_data)
        
        assert not result.is_valid
        assert "No negative examples found" in result.errors
        
    def test_invalid_confidence_validation(self):
        """Test validation of invalid confidence values."""
        calibrator = ConfidenceCalibrator()
        
        # Create valid data first, then modify it to bypass Pydantic validation
        valid_data = create_sample_calibration_data(n_samples=5)
        
        # Manually modify confidence to invalid value (bypassing Pydantic)
        valid_data[0].__dict__['heuristic_confidence'] = 1.5  # Invalid
        
        result = calibrator.validate_data(valid_data)
        
        assert not result.is_valid
        assert len(result.invalid_samples) == 1
        assert any("Invalid confidence" in error for error in result.errors)


class TestEnhancedCalibrator:
    """Test enhanced ConfidenceCalibrator functionality."""
    
    def test_calibrator_with_model_dir(self):
        """Test calibrator initialization with model directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = ConfidenceCalibrator(method="platt", model_dir=temp_dir)
            
            assert calibrator._model_dir == Path(temp_dir)
            assert calibrator._model_dir.exists()
            
    def test_fit_with_data_split(self):
        """Test fitting with custom data split configuration."""
        calibrator = ConfidenceCalibrator(method="platt")
        sample_data = create_sample_calibration_data(n_samples=100)
        split_config = DataSplit(train_size=0.7, validation_size=0.3)
        
        calibrator.fit(sample_data, data_split=split_config)
        
        assert calibrator.is_fitted
        assert calibrator.training_metrics is not None
        
    def test_fit_with_versioning(self):
        """Test fitting with model versioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = ConfidenceCalibrator(method="platt", model_dir=temp_dir)
            sample_data = create_sample_calibration_data(n_samples=100)
            
            calibrator.fit(sample_data, version="v1.0.0", description="Initial model")
            
            assert calibrator.is_fitted
            assert calibrator.get_current_version() == "v1.0.0"
            
            versions = calibrator.list_versions()
            assert len(versions) == 1
            assert versions[0].version == "v1.0.0"
            assert versions[0].description == "Initial model"
            
    def test_rollback_to_version(self):
        """Test rolling back to a previous model version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = ConfidenceCalibrator(method="platt", model_dir=temp_dir)
            sample_data = create_sample_calibration_data(n_samples=100)
            
            # Train first version
            calibrator.fit(sample_data, version="v1.0.0", description="First version")
            v1_prediction = calibrator.predict_proba(0.7)
            
            # Train second version with different method
            calibrator.method = "isotonic"
            calibrator.fit(sample_data, version="v2.0.0", description="Second version")
            v2_prediction = calibrator.predict_proba(0.7)
            
            # Rollback to first version
            calibrator.rollback_to_version("v1.0.0")
            
            assert calibrator.get_current_version() == "v1.0.0"
            assert calibrator.method == "platt"  # Should be restored
            
            # Prediction should match v1
            rollback_prediction = calibrator.predict_proba(0.7)
            assert abs(rollback_prediction - v1_prediction) < 1e-6
            
    def test_compare_versions(self):
        """Test comparing model versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = ConfidenceCalibrator(method="platt", model_dir=temp_dir)
            sample_data = create_sample_calibration_data(n_samples=100)
            
            # Train two versions
            calibrator.fit(sample_data, version="v1.0.0", description="First version")
            calibrator.fit(sample_data, version="v2.0.0", description="Second version")
            
            comparison = calibrator.compare_versions("v1.0.0", "v2.0.0")
            
            assert "version1" in comparison
            assert "version2" in comparison
            assert "metrics_comparison" in comparison
            assert comparison["version1"]["version"] == "v1.0.0"
            assert comparison["version2"]["version"] == "v2.0.0"


class TestCSVSupport:
    """Test CSV data loading and saving functionality."""
    
    def test_save_load_csv(self):
        """Test saving and loading calibration data to/from CSV."""
        sample_data = create_sample_calibration_data(n_samples=20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_data.csv"
            
            # Save data
            save_calibration_data_to_csv(sample_data, csv_path)
            assert csv_path.exists()
            
            # Load data
            loaded_data = load_calibration_data_from_csv(csv_path)
            
            assert len(loaded_data) == len(sample_data)
            
            # Compare first data point (with floating point tolerance)
            original = sample_data[0]
            loaded = loaded_data[0]
            
            assert abs(loaded.heuristic_confidence - original.heuristic_confidence) < 1e-10
            assert loaded.true_label == original.true_label
            assert loaded.verdict == original.verdict
            
    def test_csv_with_custom_features(self):
        """Test CSV loading with custom feature columns."""
        # Create a simple CSV file
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "custom_features.csv"
            
            # Create CSV with custom columns
            df = pd.DataFrame({
                'heuristic_confidence': [0.7, 0.5, 0.9],
                'true_label': [True, False, True],
                'verdict': ['Likely Real', 'Suspicious', 'Likely Real'],
                'custom_feature_1': [0.1, 0.2, 0.3],
                'custom_feature_2': [0.4, 0.5, 0.6],
                'non_feature_column': ['a', 'b', 'c']  # Should be ignored
            })
            df.to_csv(csv_path, index=False)
            
            # Load with specific feature columns
            loaded_data = load_calibration_data_from_csv(
                csv_path, 
                feature_columns=['custom_feature_1', 'custom_feature_2']
            )
            
            assert len(loaded_data) == 3
            assert 'custom_feature_1' in loaded_data[0].features
            assert 'custom_feature_2' in loaded_data[0].features
            assert 'non_feature_column' not in loaded_data[0].features
            
    def test_csv_missing_required_columns(self):
        """Test CSV loading with missing required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "invalid.csv"
            
            # Create CSV missing required columns
            df = pd.DataFrame({
                'heuristic_confidence': [0.7, 0.5],
                'true_label': [True, False]
                # Missing 'verdict' column
            })
            df.to_csv(csv_path, index=False)
            
            with pytest.raises(ValueError, match="Missing required columns"):
                load_calibration_data_from_csv(csv_path)


class TestUnifiedDataInterface:
    """Test unified data loading and saving interface."""
    
    def test_auto_format_detection(self):
        """Test automatic format detection based on file extension."""
        sample_data = create_sample_calibration_data(n_samples=10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "data.json"
            csv_path = Path(temp_dir) / "data.csv"
            
            # Save in both formats
            save_calibration_data(sample_data, json_path, format="auto")
            save_calibration_data(sample_data, csv_path, format="auto")
            
            # Load with auto-detection
            json_loaded = load_calibration_data(json_path, format="auto")
            csv_loaded = load_calibration_data(csv_path, format="auto")
            
            assert len(json_loaded) == len(sample_data)
            assert len(csv_loaded) == len(sample_data)
            
            # Data should be equivalent
            assert json_loaded[0].heuristic_confidence == csv_loaded[0].heuristic_confidence
            assert json_loaded[0].true_label == csv_loaded[0].true_label
            
    def test_explicit_format_specification(self):
        """Test explicit format specification."""
        sample_data = create_sample_calibration_data(n_samples=5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save as JSON with explicit format
            json_path = Path(temp_dir) / "data.json"
            save_calibration_data(sample_data, json_path, format="json")
            
            # Load as JSON with explicit format
            loaded_data = load_calibration_data(json_path, format="json")
            
            assert len(loaded_data) == len(sample_data)


class TestDatasetManagement:
    """Test dataset management utilities."""
    
    def test_merge_datasets(self):
        """Test merging multiple calibration datasets."""
        dataset1 = create_sample_calibration_data(n_samples=10, random_seed=42)
        dataset2 = create_sample_calibration_data(n_samples=15, random_seed=123)
        dataset3 = create_sample_calibration_data(n_samples=10, random_seed=42)  # Duplicate of dataset1
        
        merged = merge_calibration_datasets(dataset1, dataset2, dataset3)
        
        # Should have dataset1 + dataset2 (dataset3 is duplicate of dataset1)
        assert len(merged) == 25  # 10 + 15, no duplicates
        
    def test_split_calibration_data(self):
        """Test splitting calibration data."""
        sample_data = create_sample_calibration_data(n_samples=100)
        split_config = DataSplit(train_size=0.7, validation_size=0.3)
        
        splits = split_calibration_data(sample_data, split_config)
        
        assert "train" in splits
        assert "validation" in splits
        assert len(splits["train"]) == 70
        assert len(splits["validation"]) == 30
        
    def test_three_way_split(self):
        """Test three-way data splitting."""
        sample_data = create_sample_calibration_data(n_samples=100)
        split_config = DataSplit(train_size=0.6, validation_size=0.2, test_size=0.2)
        
        splits = split_calibration_data(sample_data, split_config)
        
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert len(splits["train"]) == 60
        assert len(splits["validation"]) == 20
        assert len(splits["test"]) == 20


if __name__ == "__main__":
    pytest.main([__file__])