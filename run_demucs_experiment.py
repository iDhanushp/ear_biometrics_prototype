import numpy as np
import pandas as pd
from enhanced_features import extract_dataset_features
from enhanced_train_model import EnhancedEarCanalClassifier
import os

# Set experiment parameters
data_dir = 'recordings'
results_dir = 'performance_analysis'
os.makedirs(results_dir, exist_ok=True)

# 1. Extract features with Demucs denoising
print("Extracting features with Demucs denoising...")
X, y, feature_names = extract_dataset_features(data_dir, denoise_mode='demucs')

# 2. Save features for analysis
features_df = pd.DataFrame(X, columns=feature_names)
features_df['label'] = y
features_path = os.path.join(results_dir, 'features_demucs.csv')
features_df.to_csv(features_path, index=False)
print(f"Features saved to {features_path}")

# 3. Train and evaluate model using EnhancedEarCanalClassifier
print("Training and evaluating model with Demucs-denoised features...")
classifier = EnhancedEarCanalClassifier()

# Extract and prepare data (fused mode)
X_train, X_test, y_train, y_test = classifier.extract_and_prepare_data(data_dir, feature_mode='fused')

# Feature selection
X_train_selected = classifier.feature_selection(X_train, y_train, method='rfe', k=80)
X_test_selected = classifier.feature_selector.transform(X_test)

# Train and optimize models
best_models = classifier.train_and_optimize(X_train_selected, y_train)

# Evaluate models
results, y_test_final, y_pred_final = classifier.evaluate_models(
    best_models, X_train_selected, X_test_selected, y_train, y_test
)

# Analyze results
classifier.analyze_results(results, y_test_final, y_pred_final)

# Save model
classifier.save_model(filename_prefix='enhanced_ear_biometric_demucs')

# 4. Save results
results_path = os.path.join(results_dir, 'results_demucs_summary_20250610.txt')
with open(results_path, 'w') as f:
    f.write("Enhanced Ear Canal Biometric Authentication System - Demucs Experiment\n")
    f.write("=" * 80 + "\n\n")
    f.write("Feature Extraction with Demucs Denoising\n")
    f.write(f"Total recordings: {len(y)}\n")
    f.write(f"Total features: {len(feature_names)}\n")
    f.write(f"Users: {len(set(y))}\n\n")
    f.write("Model Results:\n")
    for model_name, model_info in results.items():
        f.write(f"{model_name}: {model_info['accuracy']:.4f}\n")
    f.write(f"\nBest model accuracy: {max(results.values(), key=lambda x: x['accuracy'])['accuracy']:.4f}\n")
print(f"Results saved to {results_path}")
