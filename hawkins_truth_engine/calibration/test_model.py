"""
Unit tests for the calibration model module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hawkins_truth_engine.calibration.model import (
    CalibrationDataPoint,
    CalibrationMetrics,
    ConfidenceCalibrator,
    create_sample_calibration_data,
    load_calibration_data_from_json,
    save_calibration_data_to_json,
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


if __name__ == "__main__":
    pytest.main([__file__])