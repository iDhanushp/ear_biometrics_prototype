#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import librosa
import os
from enhanced_features import extract_ear_canal_features
from enhanced_predict import EnhancedEarCanalPredictor
import joblib
from tqdm import tqdm

class ModelAnalyzer:
    def __init__(self, recordings_dir='recordings', model_prefix='enhanced_ear_biometric'):
        """Initialize the model analyzer with recordings directory and model files."""
        self.recordings_dir = recordings_dir
        self.predictor = EnhancedEarCanalPredictor(model_prefix)
        self.model = self.predictor.model
        self.label_encoder = self.predictor.label_encoder
        self.users = self.label_encoder.classes_
        
        # Create directories for analysis output
        self.analysis_dir = 'performance_analysis'
        self.viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def analyze_all_recordings(self):
        """Analyze all recordings and generate comprehensive metrics."""
        print("Analyzing all recordings...")
        
        # Get all wav files
        wav_files = [f for f in os.listdir(self.recordings_dir) if f.endswith('.wav')]
        
        # Initialize storage for predictions and true labels
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        all_confidences = []
        
        # Process each recording
        for wav_file in tqdm(wav_files, desc="Processing recordings"):
            # Get true label from filename
            true_label = wav_file.split('_')[1]
            
            # Get prediction
            audio_path = os.path.join(self.recordings_dir, wav_file)
            predicted_user, confidence, accepted = self.predictor.predict_with_confidence(audio_path)
            
            # Get probabilities
            _, probabilities = self.predictor.predict(audio_path, return_probabilities=True)
            
            # Store results
            all_predictions.append(predicted_user)
            all_true_labels.append(true_label)
            all_confidences.append(confidence)
            all_probabilities.append(probabilities)
        
        # Convert to numpy arrays
        self.y_true = np.array(all_true_labels)
        self.y_pred = np.array(all_predictions)
        self.confidences = np.array(all_confidences)
        self.probabilities = np.array(all_probabilities)
        
        return self.generate_performance_metrics()
    
    def generate_performance_metrics(self):
        """Generate comprehensive performance metrics and visualizations."""
        print("\nGenerating performance metrics and visualizations...")
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix()
        
        # 2. ROC Curves
        self.plot_roc_curves()
        
        # 3. Confidence Distribution
        self.plot_confidence_distribution()
        
        # 4. Per-User Performance
        self.plot_per_user_performance()
        
        # 5. Feature Importance Analysis
        self.plot_feature_importance()
        
        # 6. Error Analysis
        self.analyze_errors()
        
        # 7. Generate detailed report
        self.generate_detailed_report()
        
        print("\nAnalysis complete! Check the 'performance_analysis' directory for visualizations.")
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.users,
                   yticklabels=self.users)
        plt.title('Enhanced Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self):
        """Plot and save ROC curves for each user."""
        plt.figure(figsize=(12, 8))
        
        # Binarize the output
        y_true_bin = label_binarize(self.y_true, classes=self.users)
        
        # Calculate ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, user in enumerate(self.users):
            fpr[user], tpr[user], _ = roc_curve(y_true_bin[:, i], 
                                              [p[user] for p in self.probabilities])
            roc_auc[user] = auc(fpr[user], tpr[user])
            
            plt.plot(fpr[user], tpr[user], label=f'{user} (AUC = {roc_auc[user]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each User')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot confidence score distribution."""
        plt.figure(figsize=(12, 6))
        
        # Plot confidence distribution for correct and incorrect predictions
        correct_mask = self.y_true == self.y_pred
        plt.hist(self.confidences[correct_mask], bins=20, alpha=0.5, 
                label='Correct Predictions', color='green')
        plt.hist(self.confidences[~correct_mask], bins=20, alpha=0.5, 
                label='Incorrect Predictions', color='red')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_user_performance(self):
        """Plot per-user performance metrics."""
        # Calculate metrics for each user
        metrics = []
        for user in self.users:
            user_mask = self.y_true == user
            correct = np.sum((self.y_true == self.y_pred) & user_mask)
            total = np.sum(user_mask)
            accuracy = correct / total
            avg_confidence = np.mean(self.confidences[user_mask])
            metrics.append({
                'User': user,
                'Accuracy': accuracy,
                'Average Confidence': avg_confidence,
                'Total Samples': total
            })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        sns.barplot(data=metrics_df, x='User', y='Accuracy', ax=ax1)
        ax1.set_title('Per-User Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Confidence plot
        sns.barplot(data=metrics_df, x='User', y='Average Confidence', ax=ax2)
        ax2.set_title('Per-User Average Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'per_user_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names
            feature_names = self.predictor.feature_names
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 20 Most Important Features')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_errors(self):
        """Analyze prediction errors."""
        # Find incorrect predictions
        error_mask = self.y_true != self.y_pred
        errors = pd.DataFrame({
            'True Label': self.y_true[error_mask],
            'Predicted Label': self.y_pred[error_mask],
            'Confidence': self.confidences[error_mask]
        })
        
        if len(errors) > 0:
            # Plot error distribution
            plt.figure(figsize=(12, 6))
            error_counts = errors.groupby(['True Label', 'Predicted Label']).size().unstack(fill_value=0)
            sns.heatmap(error_counts, annot=True, fmt='d', cmap='Reds')
            plt.title('Error Distribution Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save error details to CSV
            errors.to_csv(os.path.join(self.analysis_dir, 'error_details.csv'), index=False)
    
    def generate_detailed_report(self):
        """Generate a detailed performance report."""
        # Calculate overall metrics
        accuracy = np.mean(self.y_true == self.y_pred)
        avg_confidence = np.mean(self.confidences)
        
        # Generate classification report
        report = classification_report(self.y_true, self.y_pred, 
                                     target_names=self.users, 
                                     output_dict=True)
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Calculate additional metrics
        metrics = {
            'Overall Accuracy': accuracy,
            'Average Confidence': avg_confidence,
            'Total Samples': len(self.y_true),
            'Number of Users': len(self.users),
            'Error Rate': 1 - accuracy
        }
        
        # Save metrics
        with open(os.path.join(self.analysis_dir, 'detailed_metrics.txt'), 'w') as f:
            f.write("=== Enhanced Model Performance Report ===\n\n")
            f.write("Overall Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nPer-User Metrics:\n")
            f.write(report_df.to_string())
            
            f.write("\n\nConfidence Analysis:\n")
            f.write(f"Average Confidence (Correct Predictions): {np.mean(self.confidences[self.y_true == self.y_pred]):.4f}\n")
            f.write(f"Average Confidence (Incorrect Predictions): {np.mean(self.confidences[self.y_true != self.y_pred]):.4f}\n")
            
            f.write("\n\nError Analysis:\n")
            error_rate = 1 - accuracy
            f.write(f"Overall Error Rate: {error_rate:.4f}\n")
            if len(self.y_true[self.y_true != self.y_pred]) > 0:
                f.write("\nMost Common Errors:\n")
                error_counts = pd.crosstab(self.y_true[self.y_true != self.y_pred], 
                                         self.y_pred[self.y_true != self.y_pred])
                f.write(error_counts.to_string())

def main():
    """Main function to run the analysis."""
    print("Starting comprehensive model performance analysis...")
    
    analyzer = ModelAnalyzer()
    analyzer.analyze_all_recordings()
    
    print("\nAnalysis complete! Check the following directories for results:")
    print("1. performance_analysis/visualizations/ - Contains all visualization plots")
    print("2. performance_analysis/ - Contains detailed metrics and error analysis")
    print("\nGenerated files include:")
    print("- Visualizations:")
    print("  * Confusion matrix")
    print("  * ROC curves for each user")
    print("  * Confidence distribution plots")
    print("  * Per-user performance metrics")
    print("  * Feature importance analysis (if available)")
    print("  * Error distribution")
    print("- Analysis:")
    print("  * Detailed performance report")
    print("  * Error details (CSV)")

if __name__ == "__main__":
    main() 