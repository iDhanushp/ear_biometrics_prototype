#!/usr/bin/env python3
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def analyze_model():
    """Analyze and display the trained model results."""
    # Load the model and scaler
    model = joblib.load('user_identification_model.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    
    # Load and print the classification report
    try:
        y_test = np.load('y_test.npy', allow_pickle=True)
        y_pred = np.load('y_pred.npy', allow_pickle=True)
        print("\nClassification Report (from saved test set):")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print("Could not load or print classification report:", e)
    
    # Load the feature importance plot
    plt.figure(figsize=(12, 6))
    img = plt.imread('feature_importance.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Feature Importance Analysis')
    plt.show()
    
    # Load and display confusion matrix
    plt.figure(figsize=(10, 8))
    img = plt.imread('confusion_matrix.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print model information
    print("\nModel Information:")
    print(f"Number of trees in Random Forest: {model.n_estimators}")
    print(f"Number of features used: {model.n_features_in_}")
    
    # Create feature names based on the number of features
    feature_names = []
    for i in range(13):  # 13 MFCC coefficients
        feature_names.extend([f'mfcc_{i+1}_mean', f'mfcc_{i+1}_std'])
    feature_names.extend([
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'zero_crossing_rate_mean', 'zero_crossing_rate_std',
        'rms_mean', 'rms_std'
    ])
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importances.head(10).to_string(index=False))
    
    # Print class information
    print("\nClasses (Users) in the model:")
    for i, class_name in enumerate(model.classes_):
        print(f"{i+1}. {class_name}")

if __name__ == "__main__":
    analyze_model() 