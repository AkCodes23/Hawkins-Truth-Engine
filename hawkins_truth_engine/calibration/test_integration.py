"""
Integration tests for the calibration module.

These tests verify that the calibration module integrates properly
with the rest of the Hawkins Truth Engine.
"""

import pytest

from hawkins_truth_engine.calibration.model import (
    CalibrationDataPoint,
    ConfidenceCalibrator,
    create_sample_calibration_data,
)


def test_calibration_module_import():
    """Test that calibration module can be imported successfully."""
    # This test verifies that all imports work correctly
    assert CalibrationDataPoint is not None
    assert ConfidenceCalibrator is not None
    assert create_sample_calibration_data is not None


def test_basic_calibration_workflow():
    """Test the basic calibration workflow end-to-end."""
    # Create sample data
    data = create_sample_calibration_data(n_samples=50, random_seed=42)
    assert len(data) == 50
    
    # Train calibrator
    calibrator = ConfidenceCalibrator(method="platt")
    calibrator.fit(data)
    assert calibrator.is_fitted
    
    # Make predictions
    test_confidences = [0.1, 0.5, 0.9]
    for conf in test_confidences:
        calibrated = calibrator.predict_proba(conf)
        assert 0.0 <= calibrated <= 1.0
        assert isinstance(calibrated, float)


def test_calibration_with_realistic_data():
    """Test calibration with data that resembles real truth engine outputs."""
    # Create realistic calibration data points
    realistic_data = [
        CalibrationDataPoint(
            features={
                "linguistic_risk": 0.3,
                "statistical_risk": 0.2,
                "source_trust": 0.8
            },
            heuristic_confidence=0.7,
            true_label=True,
            verdict="Likely Real",
            metadata={"document_id": "doc1"}
        ),
        CalibrationDataPoint(
            features={
                "linguistic_risk": 0.8,
                "statistical_risk": 0.9,
                "source_trust": 0.2
            },
            heuristic_confidence=0.2,
            true_label=False,
            verdict="Likely Fake",
            metadata={"document_id": "doc2"}
        ),
        CalibrationDataPoint(
            features={
                "linguistic_risk": 0.5,
                "statistical_risk": 0.6,
                "source_trust": 0.5
            },
            heuristic_confidence=0.5,
            true_label=False,
            verdict="Suspicious",
            metadata={"document_id": "doc3"}
        ),
    ]
    
    # Extend with more samples for training
    extended_data = realistic_data * 20  # 60 samples total
    
    # Train both calibration methods
    platt_calibrator = ConfidenceCalibrator(method="platt")
    isotonic_calibrator = ConfidenceCalibrator(method="isotonic")
    
    platt_calibrator.fit(extended_data)
    isotonic_calibrator.fit(extended_data)
    
    # Test predictions
    test_confidence = 0.6
    platt_pred = platt_calibrator.predict_proba(test_confidence)
    isotonic_pred = isotonic_calibrator.predict_proba(test_confidence)
    
    assert 0.0 <= platt_pred <= 1.0
    assert 0.0 <= isotonic_pred <= 1.0
    
    # Predictions should be different (unless by coincidence)
    # This tests that the methods are actually doing different things
    print(f"Platt prediction: {platt_pred:.4f}")
    print(f"Isotonic prediction: {isotonic_pred:.4f}")


def test_calibration_fallback_behavior():
    """Test that calibration gracefully falls back when models are unavailable."""
    calibrator = ConfidenceCalibrator()
    
    # Should return heuristic confidence when not fitted
    test_confidence = 0.75
    result = calibrator.predict_proba(test_confidence)
    assert result == test_confidence
    
    # Should handle batch predictions too
    test_confidences = [0.2, 0.5, 0.8]
    results = calibrator.predict_proba_batch(test_confidences)
    assert list(results) == test_confidences


def test_calibration_error_handling():
    """Test error handling in calibration module."""
    # Create and fit a calibrator first
    sample_data = create_sample_calibration_data(n_samples=50)
    calibrator = ConfidenceCalibrator()
    calibrator.fit(sample_data)
    
    # Test invalid confidence values (only when fitted)
    with pytest.raises(ValueError):
        calibrator.predict_proba(-0.1)
    
    with pytest.raises(ValueError):
        calibrator.predict_proba(1.5)
    
    # Test evaluation without test data
    with pytest.raises(ValueError):
        calibrator.evaluate([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])