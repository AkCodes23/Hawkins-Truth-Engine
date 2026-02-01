# Confidence Calibration Module

This module provides confidence calibration functionality for the Hawkins Truth Engine, converting heuristic confidence scores into calibrated probabilities using machine learning techniques.

## Overview

The calibration module implements two main calibration methods:

1. **Platt Scaling**: Uses a sigmoid function to map confidence scores to probabilities
2. **Isotonic Regression**: Uses a non-parametric approach that preserves monotonicity

## Key Components

### CalibrationDataPoint

A Pydantic model representing a single training data point:

```python
CalibrationDataPoint(
    features={"linguistic_risk": 0.3, "source_trust": 0.8},
    heuristic_confidence=0.7,
    true_label=True,
    verdict="Likely Real",
    metadata={"document_id": "doc1"}
)
```

### ConfidenceCalibrator

The main calibration class that handles training and prediction:

```python
# Initialize calibrator
calibrator = ConfidenceCalibrator(method="platt")  # or "isotonic"

# Train on labeled data
calibrator.fit(training_data)

# Make predictions
calibrated_prob = calibrator.predict_proba(0.7)  # Single prediction
calibrated_probs = calibrator.predict_proba_batch([0.3, 0.5, 0.8])  # Batch

# Evaluate quality
metrics = calibrator.evaluate(test_data)
print(f"Brier Score: {metrics.brier_score}")
```

## Usage Examples

### Basic Usage

```python
from hawkins_truth_engine.calibration.model import (
    ConfidenceCalibrator,
    create_sample_calibration_data
)

# Create sample data for demonstration
train_data = create_sample_calibration_data(n_samples=200)
test_data = create_sample_calibration_data(n_samples=100, random_seed=123)

# Train calibrator
calibrator = ConfidenceCalibrator(method="platt")
calibrator.fit(train_data)

# Make predictions
original_confidence = 0.75
calibrated_confidence = calibrator.predict_proba(original_confidence)
print(f"Original: {original_confidence}, Calibrated: {calibrated_confidence}")

# Evaluate performance
metrics = calibrator.evaluate(test_data)
print(f"Brier Score: {metrics.brier_score:.4f}")
print(f"Reliability Score: {metrics.reliability_score:.4f}")
```

### Model Persistence

```python
# Save trained model
calibrator.save_model("models/calibrator.joblib")

# Load model later
new_calibrator = ConfidenceCalibrator()
new_calibrator.load_model("models/calibrator.joblib")
```

### Data Management

```python
from hawkins_truth_engine.calibration.model import (
    save_calibration_data_to_json,
    load_calibration_data_from_json
)

# Save training data
save_calibration_data_to_json(train_data, "data/calibration_data.json")

# Load training data
loaded_data = load_calibration_data_from_json("data/calibration_data.json")
```

## Integration with Truth Engine

The calibration module is designed to integrate seamlessly with the existing Hawkins Truth Engine pipeline:

1. **Data Collection**: Collect heuristic confidence scores and ground truth labels during normal operation
2. **Model Training**: Periodically train calibration models on collected data
3. **Runtime Calibration**: Apply trained models to convert heuristic confidence to calibrated probabilities
4. **Fallback Behavior**: Gracefully fall back to heuristic confidence when calibration is unavailable

## Calibration Methods

### Platt Scaling

- **Pros**: Fast, works well with small datasets, provides smooth probability estimates
- **Cons**: Assumes sigmoid relationship, may not capture complex patterns
- **Best for**: General-purpose calibration, when you have limited training data

### Isotonic Regression

- **Pros**: Non-parametric, preserves monotonicity, can capture complex relationships
- **Cons**: May overfit with small datasets, less smooth than Platt scaling
- **Best for**: When you have sufficient training data and complex calibration patterns

## Evaluation Metrics

The module provides several metrics to evaluate calibration quality:

- **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes (lower is better)
- **Log Loss**: Measures the negative log-likelihood of predictions (lower is better)
- **Reliability Score**: Measures how well predicted probabilities match actual frequencies (higher is better)

## Files

- `model.py`: Main calibration implementation
- `test_model.py`: Comprehensive unit tests
- `test_integration.py`: Integration tests
- `example.py`: Usage examples and demonstrations
- `README.md`: This documentation

## Requirements

- scikit-learn >= 1.4
- numpy >= 1.26
- pydantic >= 2.6
- joblib >= 1.3

## Testing

Run the test suite:

```bash
python -m pytest hawkins_truth_engine/calibration/ -v
```

Run the example:

```bash
python hawkins_truth_engine/calibration/example.py
```