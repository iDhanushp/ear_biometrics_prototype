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
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import unique_labels

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
        X, y, self.feature_names = extract_dataset_features(recordings_dir, feature_mode=feature_mode)
        # Debug: Print per-user sample counts before filtering
        import collections
        user_counts = collections.Counter(y)
        print("Sample count per user BEFORE filtering:", dict(user_counts))
        print("Total samples:", len(y))
        print("Unique users:", len(user_counts))
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
        # Remove classes with fewer than 2 samples (required for stratified split)
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        valid_classes = unique_classes[class_counts >= 2]
        if len(valid_classes) < len(unique_classes):
            removed = set(unique_classes) - set(valid_classes)
            print(f"[WARNING] Removed {len(removed)} user(s) with <2 recordings: {removed}")
            mask = np.isin(y_encoded, valid_classes)
            X = X[mask]
            y_encoded = y_encoded[mask]
        # Debug: Print per-user sample counts after filtering
        user_counts_after = collections.Counter(y_encoded)
        print("Sample count per user AFTER filtering:", dict(user_counts_after))
        print("Total samples after filtering:", sum(user_counts_after.values()))
        print("Unique users after filtering:", len(user_counts_after))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        # Debug: Print per-user sample counts in train and test sets
        print("Train set user counts:", dict(collections.Counter(y_train)))
        print("Test set user counts:", dict(collections.Counter(y_test)))
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_selection(self, X_train, y_train, method='rfe', k=50, cv=3):
        """Perform feature selection to reduce dimensionality. Supports 'rfe', 'rfecv', and 'univariate'."""
        print(f"\n=== Feature Selection ({method.upper()}) ===")
        if method == 'univariate':
            # Univariate feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # Recursive feature elimination with Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
        elif method == 'rfecv':
            # Recursive feature elimination with cross-validation
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFECV(estimator=rf, step=1, cv=cv, scoring='accuracy', min_features_to_select=10)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
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
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 2, 10],
                    'degree': [2, 3, 4]  # for poly kernel
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
                    'hidden_layer_sizes': [
                        (32,), (64,), (128,), (256,), (512,),
                        (50,), (100,), (100, 50), (200, 100), (256, 128, 64), (128, 128, 128),
                        (256, 128, 64, 32), (128, 128, 128, 64),
                        (512, 256, 128), (256, 256, 128, 64), (128, 64, 32), (512, 256, 128, 64)
                    ],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                    'learning_rate': ['constant', 'adaptive', 'invscaling'],
                    'batch_size': ['auto', 8, 16, 32, 64],
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
        
        # Only use user_names for classes present in y_test/y_pred
        user_names = self.label_encoder.classes_
        present_labels = unique_labels(y_test, y_pred)
        present_user_names = [user_names[i] for i in present_labels]
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_user_names))
        
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

    def predict_with_voting(self, X_samples, window=3, voting='hard'):
        """
        Predict user label from multiple consecutive feature vectors using voting.
        Args:
            X_samples: np.ndarray of shape (n_samples, n_features) - consecutive recordings for a single authentication attempt
            window: Number of samples to consider for voting (default: 3)
            voting: 'hard' (majority vote) or 'soft' (average probabilities)
        Returns:
            predicted_label: predicted user label (decoded)
            votes: list of predicted labels or probabilities
        """
        # Preprocess
        X_scaled = self.scaler.transform(X_samples)
        if self.feature_selector:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        if voting == 'hard':
            preds = self.best_model.predict(X_scaled)
            # Majority vote
            from collections import Counter
            vote_counts = Counter(preds)
            majority_label = vote_counts.most_common(1)[0][0]
            decoded_label = self.label_encoder.inverse_transform([majority_label])[0]
            return decoded_label, preds
        elif voting == 'soft' and hasattr(self.best_model, 'predict_proba'):
            probas = self.best_model.predict_proba(X_scaled)
            # Average probabilities over the voting window
            if window > 1 and len(probas) >= window:
                probas = np.mean(probas[-window:], axis=0)
            else:
                probas = np.mean(probas, axis=0)
            best_idx = np.argmax(probas)
            decoded_label = self.label_encoder.inverse_transform([best_idx])[0]
            return decoded_label, probas
        else:
            raise ValueError("Voting must be 'hard' or 'soft' and model must support predict_proba for soft voting.")

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
    import argparse
    parser = argparse.ArgumentParser(
        description='Enhanced Ear Canal Biometric Training',
        epilog='''\nTest-time voting options:\n  --voting: Enable prediction with voting over multiple samples.\n  --voting_window: Number of consecutive samples to use for voting (default: 3).\n  --voting_type: Voting method: majority or soft (default: majority).\n\nExample: python enhanced_train_model.py --feature_mode voice --voting --voting_window 5 --voting_type soft\n''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--feature_mode', choices=['fused', 'echo', 'voice'], default='fused',
                       help='Feature mode: fused (default), echo, or voice')
    parser.add_argument('--recordings_dir', default='recordings', 
                       help='Directory containing recordings')
    parser.add_argument('--feature_selection', choices=['rfe', 'rfecv', 'univariate'], default='rfe',
                       help='Feature selection method: rfe (default), rfecv, or univariate')
    parser.add_argument('--num_features', type=int, default=None,
                       help='Number of features to select (ignored for rfecv)')
    # Voting CLI arguments
    parser.add_argument('--voting', action='store_true',
                       help='Enable test-time voting for prediction (overrides normal evaluation)')
    parser.add_argument('--voting_window', type=int, default=3,
                       help='Number of consecutive samples to use for voting (default: 3)')
    parser.add_argument('--voting_type', choices=['majority', 'soft'], default='majority',
                       help='Voting method: majority or soft (default: majority)')
    parser.add_argument('--voting_input', type=str, default=None,
                       help='CSV file with features for voting prediction (optional)')
    args = parser.parse_args()
    print("Enhanced Ear Canal Biometric Authentication System")
    print("=" * 60)
    print(f"Feature mode: {args.feature_mode}")
    classifier = EnhancedEarCanalClassifier()
    X_train, X_test, y_train, y_test = classifier.extract_and_prepare_data(
        recordings_dir=args.recordings_dir, 
        feature_mode=args.feature_mode
    )
    if X_train.shape[1] > 0:
        if args.feature_selection == 'rfecv':
            X_train_sel = classifier.feature_selection(X_train, y_train, method='rfecv', cv=3)
            X_test_sel = classifier.feature_selector.transform(X_test)
        else:
            k_features = args.num_features if args.num_features is not None else min(50, X_train.shape[1] // 2)
            X_train_sel = classifier.feature_selection(X_train, y_train, method=args.feature_selection, k=k_features)
            X_test_sel = classifier.feature_selector.transform(X_test)
        print(f"\nTraining {args.feature_mode} model...")
        best_models = classifier.train_and_optimize(X_train_sel, y_train)
        print(f"\nEvaluating {args.feature_mode} model...")
        results, y_test_final, y_pred_final = classifier.evaluate_models(
            best_models, X_train_sel, X_test_sel, y_train, y_test
        )
        classifier.save_model(filename_prefix=f'enhanced_ear_biometric_{args.feature_mode}')
        classifier.analyze_results(results, y_test_final, y_pred_final)
        print(f"\n{args.feature_mode.capitalize()} model training complete!")
        # Test-time voting CLI
        if args.voting:
            print("\n=== Test-Time Voting Prediction ===")
            if args.voting_input:
                # Load features from CSV for voting
                df = pd.read_csv(args.voting_input)
                X_vote = df.values
            else:
                # Use X_test_sel as default
                X_vote = X_test_sel
            # Use best model for voting
            y_vote_pred = classifier.predict_with_voting(
                X_vote,
                window=args.voting_window,
                voting=args.voting_type
            )
            print(f"Voting predictions (window={args.voting_window}, type={args.voting_type}):\n{y_vote_pred}")
    else:
        print(f"ERROR: No {args.feature_mode} features found!")
        print("This might be because:")
        print("1. Your recordings don't have the required metadata for voice features")
        print("2. The feature extraction didn't properly classify the modality")
        print("3. The recordings need proper 'phrase' metadata for voice classification")

if __name__ == "__main__":
    main()

