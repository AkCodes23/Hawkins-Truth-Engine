"""
Example usage of the confidence calibration module.

This script demonstrates how to use the ConfidenceCalibrator class
to train and use calibration models.
"""

import tempfile
from pathlib import Path

from hawkins_truth_engine.calibration.model import (
    ConfidenceCalibrator,
    create_sample_calibration_data,
    save_calibration_data_to_json,
    load_calibration_data_from_json,
)


def main():
    """Demonstrate calibration functionality."""
    print("=== Hawkins Truth Engine - Confidence Calibration Example ===\n")
    
    # 1. Create sample calibration data
    print("1. Creating sample calibration data...")
    train_data = create_sample_calibration_data(n_samples=200, random_seed=42)
    test_data = create_sample_calibration_data(n_samples=100, random_seed=123)
    print(f"   Created {len(train_data)} training samples and {len(test_data)} test samples")
    
    # 2. Save and load data (demonstrate data management)
    print("\n2. Demonstrating data persistence...")
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "calibration_data.json"
        save_calibration_data_to_json(train_data[:10], data_path)
        loaded_data = load_calibration_data_from_json(data_path)
        print(f"   Saved and loaded {len(loaded_data)} data points successfully")
    
    # 3. Train Platt scaling calibrator
    print("\n3. Training Platt scaling calibrator...")
    platt_calibrator = ConfidenceCalibrator(method="platt")
    platt_calibrator.fit(train_data)
    print("   Platt scaling calibrator trained successfully")
    
    # 4. Train isotonic regression calibrator
    print("\n4. Training isotonic regression calibrator...")
    isotonic_calibrator = ConfidenceCalibrator(method="isotonic")
    isotonic_calibrator.fit(train_data)
    print("   Isotonic regression calibrator trained successfully")
    
    # 5. Compare predictions
    print("\n5. Comparing calibration methods...")
    test_confidences = [0.2, 0.4, 0.6, 0.8, 0.9]
    
    print("   Heuristic -> Platt -> Isotonic")
    for conf in test_confidences:
        platt_pred = platt_calibrator.predict_proba(conf)
        isotonic_pred = isotonic_calibrator.predict_proba(conf)
        print(f"   {conf:.1f}       -> {platt_pred:.3f} -> {isotonic_pred:.3f}")
    
    # 6. Evaluate calibrators
    print("\n6. Evaluating calibration quality...")
    platt_metrics = platt_calibrator.evaluate(test_data)
    isotonic_metrics = isotonic_calibrator.evaluate(test_data)
    
    print(f"   Platt Scaling:")
    print(f"     Brier Score: {platt_metrics.brier_score:.4f}")
    print(f"     Log Loss: {platt_metrics.log_loss:.4f}")
    print(f"     Reliability Score: {platt_metrics.reliability_score:.4f}")
    
    print(f"   Isotonic Regression:")
    print(f"     Brier Score: {isotonic_metrics.brier_score:.4f}")
    print(f"     Log Loss: {isotonic_metrics.log_loss:.4f}")
    print(f"     Reliability Score: {isotonic_metrics.reliability_score:.4f}")
    
    # 7. Demonstrate model persistence
    print("\n7. Demonstrating model persistence...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "platt_calibrator.joblib"
        
        # Save model
        platt_calibrator.save_model(model_path)
        print(f"   Model saved to {model_path}")
        
        # Load model into new calibrator
        new_calibrator = ConfidenceCalibrator()
        new_calibrator.load_model(model_path)
        print("   Model loaded successfully")
        
        # Verify predictions are the same
        original_pred = platt_calibrator.predict_proba(0.7)
        loaded_pred = new_calibrator.predict_proba(0.7)
        print(f"   Original prediction: {original_pred:.6f}")
        print(f"   Loaded prediction: {loaded_pred:.6f}")
        print(f"   Difference: {abs(original_pred - loaded_pred):.8f}")
    
    # 8. Show model information
    print("\n8. Model information...")
    platt_info = platt_calibrator.get_model_info()
    print(f"   Method: {platt_info['method']}")
    print(f"   Is fitted: {platt_info['is_fitted']}")
    if platt_info['training_metrics']:
        print(f"   Training Brier Score: {platt_info['training_metrics']['brier_score']:.4f}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()