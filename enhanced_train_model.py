#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_class_weight

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Import our enhanced feature extraction
from enhanced_features import extract_dataset_features

class EnhancedEarCanalClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        
    def extract_and_prepare_data(self, recordings_dir='recordings', test_size=0.2, feature_mode='fused'):
        """
        Extract features and prepare train/test sets.
        feature_mode: 'fused', 'echo', or 'voice'
        """
        print(f"=== Feature Extraction ({feature_mode}) ===")
        X, y, self.feature_names = extract_dataset_features(recordings_dir)
        # Filter features by mode
        if feature_mode == 'echo':
            echo_cols = [i for i, name in enumerate(self.feature_names) if name.startswith('echo_')]
            X = X[:, echo_cols]
            self.feature_names = [self.feature_names[i] for i in echo_cols]
        elif feature_mode == 'voice':
            voice_cols = [i for i, name in enumerate(self.feature_names) if name.startswith('voice_')]
            X = X[:, voice_cols]
            self.feature_names = [self.feature_names[i] for i in voice_cols]
        # else: fused (all features)
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        print(f"Number of users: {len(np.unique(y_encoded))}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_selection(self, X_train, y_train, method='rfe', k=50):
        """Perform feature selection to reduce dimensionality."""
        print(f"\n=== Feature Selection ({method.upper()}) ===")
        
        if method == 'univariate':
            # Univariate feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # Recursive feature elimination with Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_features = [self.feature_names[i] for i, selected in 
                               enumerate(self.feature_selector.get_support()) if selected]
            print(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")
            print("\nTop 10 selected features:")
            for i, feature in enumerate(selected_features[:10]):
                print(f"{i+1}. {feature}")
        
        return X_train_selected
    
    def define_models(self):
        """Define multiple ML models with hyperparameter grids."""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.01, 0.05, 0.1, 0.5, 2]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (256, 128, 64), (128, 128, 128)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'alpha': [1e-5, 0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.0005, 0.001, 0.005],
                    'learning_rate': ['constant', 'adaptive'],
                    'batch_size': ['auto', 32, 64],
                    'max_iter': [3000],
                    'early_stopping': [True]
                }
            }
        }
    
    def train_and_optimize(self, X_train, y_train, cv_folds=3):
        """Train multiple models with hyperparameter optimization using GridSearchCV for all."""
        print("\n=== Model Training and Optimization ===")
        self.define_models()
        best_models = {}
        
        for model_name, model_config in self.models.items():
            print(f"\nTraining {model_name}...")
            
            search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            best_models[model_name] = {
                'model': search.best_estimator_,
                'best_params': search.best_params_,
                'cv_score': search.best_score_
            }
            
            print(f"Best CV score: {search.best_score_:.4f}")
            print(f"Best params: {search.best_params_}")
        
        return best_models
    
    def create_ensemble(self, best_models):
        """Create an ensemble of the best models."""
        print("\n=== Creating Ensemble Model ===")
        
        # Select top 3 models based on CV score
        sorted_models = sorted(best_models.items(), 
                             key=lambda x: x[1]['cv_score'], reverse=True)
        
        top_models = []
        for model_name, model_info in sorted_models[:3]:
            top_models.append((model_name, model_info['model']))
            print(f"Including {model_name} (CV: {model_info['cv_score']:.4f})")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'  # Use probabilities for voting
        )
        
        return ensemble
    
    def evaluate_models(self, models, X_train, X_test, y_train, y_test):
        """Evaluate all models and select the best one."""
        print("\n=== Model Evaluation ===")
        
        results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'cv_score': model_info['cv_score']
            }
            
            print(f"{model_name:20} - Test Accuracy: {accuracy:.4f}")
        
        # Evaluate ensemble
        print("\nTraining ensemble...")
        ensemble = self.create_ensemble(models)
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        results['ensemble'] = {
            'model': ensemble,
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred,
            'cv_score': ensemble_accuracy  # Use test accuracy as proxy
        }
        
        print(f"{'ensemble':20} - Test Accuracy: {ensemble_accuracy:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        return results, y_test, results[best_model_name]['predictions']
    
    def analyze_results(self, results, y_test, y_pred):
        """Create detailed analysis and visualizations."""
        print("\n=== Results Analysis ===")
        
        # Classification report
        user_names = self.label_encoder.classes_
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=user_names))
        
        # Confusion matrix
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=user_names, yticklabels=user_names)
        plt.title('Enhanced Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_score'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test accuracies
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # CV scores
        bars2 = ax2.bar(model_names, cv_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Cross-Validation Score Comparison')
        ax2.set_ylabel('CV Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars2, cv_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        if hasattr(self.best_model, 'feature_importances_'):
            # Get selected feature names
            if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                selected_features = [self.feature_names[i] for i, selected in 
                                   enumerate(self.feature_selector.get_support()) if selected]
            else:
                selected_features = self.feature_names
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 20 Most Important Features (Enhanced Model)')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:30} ({row['importance']:.4f})")
    
    def save_model(self, filename_prefix='enhanced_ear_biometric'):
        """Save the trained model and preprocessing components."""
        joblib.dump(self.best_model, f'{filename_prefix}_model.joblib')
        joblib.dump(self.scaler, f'{filename_prefix}_scaler.joblib')
        joblib.dump(self.label_encoder, f'{filename_prefix}_label_encoder.joblib')
        
        if self.feature_selector:
            joblib.dump(self.feature_selector, f'{filename_prefix}_feature_selector.joblib')
        
        # Save feature names
        with open(f'{filename_prefix}_feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        print(f"\nModel saved with prefix: {filename_prefix}")
    
    def split_modalities(self, X, feature_names):
        """Split features into echo-only and voice-only based on feature name prefixes."""
        echo_indices = [i for i, name in enumerate(feature_names) if name.startswith('echo_')]
        voice_indices = [i for i, name in enumerate(feature_names) if name.startswith('voice_')]
        X_echo = X[:, echo_indices]
        X_voice = X[:, voice_indices]
        echo_names = [feature_names[i] for i in echo_indices]
        voice_names = [feature_names[i] for i in voice_indices]
        # Print feature counts for debug
        print(f"Echo-only features: {len(echo_names)}")
        print(f"Voice-only features: {len(voice_names)}")
        return X_echo, echo_names, X_voice, voice_names

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_set_name):
    # Classical models (SVM, RF, etc.)
    # ...existing code...

    # Neural Network (MLP) - Improved settings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights for imbalanced data
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    mlp_param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (256, 128, 64)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.0005],
        'max_iter': [3000],
        'early_stopping': [True],
        'random_state': [42]
    }
    from sklearn.model_selection import GridSearchCV
    mlp = MLPClassifier()
    mlp_grid = GridSearchCV(mlp, mlp_param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    mlp_grid.fit(X_train_scaled, y_train)
    best_mlp = mlp_grid.best_estimator_
    val_acc = best_mlp.score(X_val_scaled, y_val)
    test_acc = best_mlp.score(X_test_scaled, y_test)
    print(f"[MLP] {feature_set_name} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f} | Best Params: {mlp_grid.best_params_}")
    # ...existing code...

def main():
    print("Enhanced Ear Canal Biometric Authentication System")
    print("=" * 60)
    classifier = EnhancedEarCanalClassifier()
    # Extract all features ONCE
    X, y, feature_names = extract_dataset_features('recordings')
    # Fused (main)
    y_encoded = classifier.label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    classifier.feature_names = feature_names
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    # Split into echo-only and voice-only (STRICT: only use correct prefixes)
    echo_indices = [i for i, name in enumerate(feature_names) if name.startswith('echo_')]
    voice_indices = [i for i, name in enumerate(feature_names) if name.startswith('voice_')]
    X_echo = X[:, echo_indices]
    echo_names = [feature_names[i] for i in echo_indices]
    X_voice = X[:, voice_indices]
    voice_names = [feature_names[i] for i in voice_indices]
    # Print feature counts for debug
    print(f"Echo-only features: {len(echo_names)}")
    print(f"Voice-only features: {len(voice_names)}")
    # Split echo
    X_echo_train, X_echo_test, y_echo_train, y_echo_test = train_test_split(
        X_echo, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler_echo = StandardScaler().fit(X_echo_train)
    X_echo_train_scaled = scaler_echo.transform(X_echo_train)
    X_echo_test_scaled = scaler_echo.transform(X_echo_test)
    # Split voice
    X_voice_train, X_voice_test, y_voice_train, y_voice_test = train_test_split(
        X_voice, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler_voice = StandardScaler().fit(X_voice_train)
    X_voice_train_scaled = scaler_voice.transform(X_voice_train)
    X_voice_test_scaled = scaler_voice.transform(X_voice_test)
    # Feature selection (RFE, k=80 for all)
    print("\n[Echo-only] Feature selection...")
    selector_echo = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=40)
    X_echo_train_sel = selector_echo.fit_transform(X_echo_train_scaled, y_echo_train)
    X_echo_test_sel = selector_echo.transform(X_echo_test_scaled)
    print("[Voice-only] Feature selection...")
    selector_voice = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=40)
    X_voice_train_sel = selector_voice.fit_transform(X_voice_train_scaled, y_voice_train)
    X_voice_test_sel = selector_voice.transform(X_voice_test_scaled)
    print("[Fused] Feature selection...")
    X_train_sel = classifier.feature_selection(X_train_scaled, y_train, method='rfe', k=80)
    X_test_sel = classifier.feature_selector.transform(X_test_scaled)
    # Train and optimize models for each modality
    print("\n[Echo-only] Training...")
    best_echo = classifier.train_and_optimize(X_echo_train_sel, y_echo_train)
    print("[Voice-only] Training...")
    best_voice = classifier.train_and_optimize(X_voice_train_sel, y_voice_train)
    print("[Fused] Training...")
    best_fused = classifier.train_and_optimize(X_train_sel, y_train)
    # Evaluate and save each
    print("\n[Echo-only] Evaluation...")
    results_echo, y_echo_test_final, y_echo_pred = classifier.evaluate_models(best_echo, X_echo_train_sel, X_echo_test_sel, y_echo_train, y_echo_test)
    classifier.save_model(filename_prefix='enhanced_ear_biometric_echo')
    print("[Voice-only] Evaluation...")
    results_voice, y_voice_test_final, y_voice_pred = classifier.evaluate_models(best_voice, X_voice_train_sel, X_voice_test_sel, y_voice_train, y_voice_test)
    classifier.save_model(filename_prefix='enhanced_ear_biometric_voice')
    print("[Fused] Evaluation...")
    results_fused, y_test_final, y_pred_final = classifier.evaluate_models(best_fused, X_train_sel, X_test_sel, y_train, y_test)
    classifier.save_model(filename_prefix='enhanced_ear_biometric')
    # Analyze results for fused (main)
    classifier.analyze_results(results_fused, y_test_final, y_pred_final)
    print("\n" + "=" * 60)
    print("Enhanced training complete! Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main()