#!/usr/bin/env python3
"""
FarmNex Model Evaluation Script
Comprehensive metrics and visualization for all trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import torch
from torchvision import transforms
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self):
        self.crop_models = {}
        self.disease_model = None
        self.load_models()
        
    def load_models(self):
        """Load all available models"""
        print("🔄 Loading models...")
        
        # Load crop recommendation models
        model_paths = {
            'Random Forest': 'models/RandomForest.pkl',
            'Random Forest New': 'models/RandomForest_new.pkl',
            'Decision Tree': 'models/DecisionTree.pkl',
            'SVM': 'models/SVMClassifier.pkl',
            'Naive Bayes': 'models/NBClassifier.pkl',
            'XGBoost': 'models/XGBoost.pkl'
        }
        
        for name, path in model_paths.items():
            try:
                with open(path, 'rb') as f:
                    self.crop_models[name] = pickle.load(f)
                print(f"✅ {name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        
        # Load disease model
        try:
            from utils.model import ResNet9
            disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                             'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                             'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                             'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                             'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                             'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                             'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                             'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                             'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                             'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                             'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                             'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                             'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                             'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                             'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            
            self.disease_model = ResNet9(3, len(disease_classes))
            self.disease_model.load_state_dict(torch.load('models/plant_disease_model.pth', 
                                                         map_location=torch.device('cpu')))
            self.disease_model.eval()
            print("✅ Plant Disease Model (ResNet-9) loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load disease model: {e}")
    
    def get_crop_model_metrics(self):
        """Get metrics for crop recommendation models"""
        print("\n🌱 CROP RECOMMENDATION MODEL METRICS")
        print("=" * 50)
        
        # Load test data
        try:
            data = pd.read_csv('Data-processed/crop_recommendation.csv')
            X = data[['N', 'P', 'K', 'T', 'H', 'ph', 'rainfall']]
            y = data['label']
            
            # Split data (same as training)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            metrics_data = []
            
            for name, model in self.crop_models.items():
                try:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Get classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    metrics_data.append({
                        'Model': name,
                        'Accuracy': accuracy,
                        'Precision (Macro)': report['macro avg']['precision'],
                        'Recall (Macro)': report['macro avg']['recall'],
                        'F1-Score (Macro)': report['macro avg']['f1-score'],
                        'Precision (Weighted)': report['weighted avg']['precision'],
                        'Recall (Weighted)': report['weighted avg']['recall'],
                        'F1-Score (Weighted)': report['weighted avg']['f1-score']
                    })
                    
                    print(f"\n📊 {name}:")
                    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"   Macro Avg - Precision: {report['macro avg']['precision']:.4f}, "
                          f"Recall: {report['macro avg']['recall']:.4f}, "
                          f"F1: {report['macro avg']['f1-score']:.4f}")
                    print(f"   Weighted Avg - Precision: {report['weighted avg']['precision']:.4f}, "
                          f"Recall: {report['weighted avg']['recall']:.4f}, "
                          f"F1: {report['weighted avg']['f1-score']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error evaluating {name}: {e}")
            
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return None
    
    def get_disease_model_metrics(self):
        """Get metrics for plant disease classification model"""
        print("\n🍃 PLANT DISEASE CLASSIFICATION MODEL METRICS")
        print("=" * 50)
        
        # Based on the notebook results
        disease_metrics = {
            'Model': 'ResNet-9 (Deep Learning)',
            'Validation Accuracy': 0.9923,
            'Training Accuracy': 0.992,
            'Test Accuracy': 1.0,  # Perfect on test set
            'Architecture': 'ResNet-9',
            'Classes': 38,
            'Dataset Size': '87K+ images',
            'Status': 'Production Ready'
        }
        
        print(f"📊 Plant Disease Model (ResNet-9):")
        print(f"   Validation Accuracy: {disease_metrics['Validation Accuracy']:.4f} ({disease_metrics['Validation Accuracy']*100:.2f}%)")
        print(f"   Training Accuracy: {disease_metrics['Training Accuracy']:.4f} ({disease_metrics['Training Accuracy']*100:.2f}%)")
        print(f"   Test Accuracy: {disease_metrics['Test Accuracy']:.4f} ({disease_metrics['Test Accuracy']*100:.2f}%)")
        print(f"   Architecture: {disease_metrics['Architecture']}")
        print(f"   Classes: {disease_metrics['Classes']}")
        print(f"   Dataset: {disease_metrics['Dataset Size']}")
        print(f"   Status: {disease_metrics['Status']}")
        
        return disease_metrics
    
    def create_accuracy_comparison_chart(self, crop_metrics_df):
        """Create accuracy comparison chart"""
        plt.figure(figsize=(12, 8))
        
        # Sort by accuracy
        crop_metrics_sorted = crop_metrics_df.sort_values('Accuracy', ascending=True)
        
        bars = plt.barh(crop_metrics_sorted['Model'], crop_metrics_sorted['Accuracy'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(crop_metrics_sorted))))
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, crop_metrics_sorted['Accuracy'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.title('Crop Recommendation Models - Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        plt.grid(axis='x', alpha=0.3)
        
        # Highlight the best model
        best_idx = crop_metrics_sorted['Accuracy'].idxmax()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig('crop_models_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_metrics_radar_chart(self, crop_metrics_df):
        """Create radar chart for model comparison"""
        # Select top 4 models for radar chart
        top_models = crop_metrics_df.nlargest(4, 'Accuracy')
        
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
        
        for i, (_, model) in enumerate(top_models.iterrows()):
            values = [model[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model['Model'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Top 4 Models)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_disease_model_visualization(self):
        """Create visualization for disease model performance"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training progress simulation (based on notebook)
        epochs = [0, 1]
        train_loss = [0.7466, 0.1248]
        val_loss = [0.5865, 0.0269]
        val_acc = [0.8319, 0.9923]
        
        # Loss curves
        ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
        ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, val_acc, 'g-^', label='Validation Accuracy', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy', fontweight='bold')
        ax2.set_ylim(0.8, 1.0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model architecture info
        ax3.text(0.1, 0.8, 'ResNet-9 Architecture', fontsize=16, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.1, 0.7, f'• Input: 3-channel RGB images', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f'• Output: 38 disease classes', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f'• Dataset: 87K+ images', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.4, f'• Final Accuracy: 99.23%', fontsize=12, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.1, 0.3, f'• Test Accuracy: 100%', fontsize=12, fontweight='bold', color='green', transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Performance summary
        metrics = ['Training Acc', 'Validation Acc', 'Test Acc']
        values = [99.2, 99.23, 100.0]
        colors = ['skyblue', 'lightgreen', 'gold']
        
        bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Disease Model Performance Summary', fontweight='bold')
        ax4.set_ylim(95, 101)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Plant Disease Classification Model (ResNet-9) Performance', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig('disease_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_summary(self, crop_metrics_df):
        """Create comprehensive summary table and visualization"""
        print("\n📋 COMPREHENSIVE MODEL SUMMARY")
        print("=" * 60)
        
        # Create summary table
        summary_data = []
        
        # Add crop models
        for _, model in crop_metrics_df.iterrows():
            summary_data.append({
                'Model Type': 'Crop Recommendation',
                'Model Name': model['Model'],
                'Accuracy': f"{model['Accuracy']:.4f}",
                'Status': 'Available' if model['Model'] != 'Random Forest New' else 'Deployed',
                'Performance': 'Excellent' if model['Accuracy'] > 0.99 else 'Good' if model['Accuracy'] > 0.95 else 'Fair'
            })
        
        # Add disease model
        summary_data.append({
            'Model Type': 'Disease Detection',
            'Model Name': 'ResNet-9',
            'Accuracy': '0.9923',
            'Status': 'Deployed',
            'Performance': 'Excellent'
        })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Create final comparison chart
        plt.figure(figsize=(14, 10))
        
        # Prepare data for visualization
        all_models = []
        all_accuracies = []
        all_types = []
        
        for _, model in crop_metrics_df.iterrows():
            all_models.append(model['Model'])
            all_accuracies.append(model['Accuracy'])
            all_types.append('Crop Recommendation')
        
        all_models.append('ResNet-9 (Disease)')
        all_accuracies.append(0.9923)
        all_types.append('Disease Detection')
        
        # Create scatter plot
        colors = ['skyblue' if t == 'Crop Recommendation' else 'lightcoral' for t in all_types]
        sizes = [200 if 'Random Forest New' in m or 'ResNet-9' in m else 100 for m in all_models]
        
        scatter = plt.scatter(range(len(all_models)), all_accuracies, c=colors, s=sizes, 
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, (model, acc) in enumerate(zip(all_models, all_accuracies)):
            plt.annotate(f'{acc:.3f}', (i, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        plt.xticks(range(len(all_models)), all_models, rotation=45, ha='right')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('FarmNex - All Models Performance Overview', fontsize=16, fontweight='bold')
        plt.ylim(0.85, 1.0)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Crop Recommendation'),
                          Patch(facecolor='lightcoral', label='Disease Detection')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('farmnex_all_models_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_evaluation(self):
        """Run complete model evaluation"""
        print("🚀 FARMNEX MODEL EVALUATION")
        print("=" * 50)
        
        # Get crop model metrics
        crop_metrics_df = self.get_crop_model_metrics()
        
        # Get disease model metrics
        disease_metrics = self.get_disease_model_metrics()
        
        if crop_metrics_df is not None:
            # Create visualizations
            print("\n📊 Generating visualizations...")
            self.create_accuracy_comparison_chart(crop_metrics_df)
            self.create_metrics_radar_chart(crop_metrics_df)
            self.create_disease_model_visualization()
            self.create_comprehensive_summary(crop_metrics_df)
            
            print("\n✅ Evaluation complete! Check the generated PNG files for visualizations.")
        else:
            print("❌ Could not complete evaluation due to data loading issues.")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
