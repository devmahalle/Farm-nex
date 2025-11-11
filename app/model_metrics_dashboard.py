#!/usr/bin/env python3
"""
FarmNex Model Metrics Dashboard
Web-based dashboard to display model performance metrics
"""

from flask import Flask, render_template_string
import pandas as pd
import json

app = Flask(__name__)

# Model metrics data (from evaluation results)
MODEL_METRICS = {
    "crop_models": [
        {
            "name": "Random Forest New",
            "accuracy": 0.9932,
            "precision_macro": 0.9926,
            "recall_macro": 0.9933,
            "f1_macro": 0.9926,
            "precision_weighted": 0.9937,
            "recall_weighted": 0.9932,
            "f1_weighted": 0.9932,
            "status": "Deployed",
            "performance": "Excellent"
        },
        {
            "name": "Decision Tree",
            "accuracy": 0.9545,
            "precision_macro": 0.9521,
            "recall_macro": 0.9548,
            "f1_macro": 0.9520,
            "precision_weighted": 0.9564,
            "recall_weighted": 0.9545,
            "f1_weighted": 0.9542,
            "status": "Available",
            "performance": "Good"
        },
        {
            "name": "SVM",
            "accuracy": 0.1159,
            "precision_macro": 0.0799,
            "recall_macro": 0.1056,
            "f1_macro": 0.0408,
            "precision_weighted": 0.0954,
            "recall_weighted": 0.1159,
            "f1_weighted": 0.0483,
            "status": "Available",
            "performance": "Poor"
        }
    ],
    "disease_model": {
        "name": "ResNet-9",
        "validation_accuracy": 0.9923,
        "training_accuracy": 0.992,
        "test_accuracy": 1.0,
        "architecture": "ResNet-9",
        "classes": 38,
        "dataset_size": "87K+ images",
        "status": "Deployed",
        "performance": "Excellent"
    }
}

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FarmNex - Model Performance Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .model-card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .model-card:hover {
            transform: translateY(-5px);
        }
        .status-badge {
            font-size: 0.9rem;
            padding: 8px 16px;
            border-radius: 20px;
        }
        .performance-excellent { background-color: #28a745; }
        .performance-good { background-color: #ffc107; color: #000; }
        .performance-poor { background-color: #dc3545; }
        .status-deployed { background-color: #007bff; }
        .status-available { background-color: #6c757d; }
    </style>
</head>
<body class="bg-light">
    <div class="container-fluid py-4">
        <div class="row">
            <div class="col-12">
                <h1 class="text-center mb-4">
                    <i class="fas fa-seedling text-success"></i>
                    FarmNex Model Performance Dashboard
                </h1>
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <i class="fas fa-brain fa-2x mb-3"></i>
                    <div class="metric-value">{{ crop_models|length }}</div>
                    <div class="metric-label">Crop Models</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <i class="fas fa-microscope fa-2x mb-3"></i>
                    <div class="metric-value">1</div>
                    <div class="metric-label">Disease Model</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <i class="fas fa-trophy fa-2x mb-3"></i>
                    <div class="metric-value">99.32%</div>
                    <div class="metric-label">Best Accuracy</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <i class="fas fa-check-circle fa-2x mb-3"></i>
                    <div class="metric-value">2</div>
                    <div class="metric-label">Deployed Models</div>
                </div>
            </div>
        </div>

        <!-- Crop Recommendation Models -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="mb-3">
                    <i class="fas fa-seedling text-primary"></i>
                    Crop Recommendation Models
                </h2>
            </div>
        </div>

        <div class="row mb-4">
            {% for model in crop_models %}
            <div class="col-md-4 mb-3">
                <div class="card model-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ model.name }}</h5>
                        <div class="mb-3">
                            <span class="badge status-badge status-{{ model.status.lower() }}">
                                {{ model.status }}
                            </span>
                            <span class="badge status-badge performance-{{ model.performance.lower() }}">
                                {{ model.performance }}
                            </span>
                        </div>
                        <div class="row text-center">
                            <div class="col-6">
                                <div class="metric-value" style="font-size: 1.8rem; color: #007bff;">
                                    {{ "%.2f"|format(model.accuracy * 100) }}%
                                </div>
                                <div class="metric-label" style="color: #6c757d;">Accuracy</div>
                            </div>
                            <div class="col-6">
                                <div class="metric-value" style="font-size: 1.8rem; color: #28a745;">
                                    {{ "%.3f"|format(model.f1_macro) }}
                                </div>
                                <div class="metric-label" style="color: #6c757d;">F1-Score</div>
                            </div>
                        </div>
                        <hr>
                        <div class="row text-center">
                            <div class="col-4">
                                <small class="text-muted">Precision</small><br>
                                <strong>{{ "%.3f"|format(model.precision_macro) }}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted">Recall</small><br>
                                <strong>{{ "%.3f"|format(model.recall_macro) }}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted">Weighted F1</small><br>
                                <strong>{{ "%.3f"|format(model.f1_weighted) }}</strong>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Disease Detection Model -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="mb-3">
                    <i class="fas fa-microscope text-danger"></i>
                    Plant Disease Detection Model
                </h2>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-8">
                <div class="card model-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ disease_model.name }} (Deep Learning)</h5>
                        <div class="mb-3">
                            <span class="badge status-badge status-{{ disease_model.status.lower() }}">
                                {{ disease_model.status }}
                            </span>
                            <span class="badge status-badge performance-{{ disease_model.performance.lower() }}">
                                {{ disease_model.performance }}
                            </span>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Architecture Details</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Type:</strong> {{ disease_model.architecture }}</li>
                                    <li><strong>Classes:</strong> {{ disease_model.classes }}</li>
                                    <li><strong>Dataset:</strong> {{ disease_model.dataset_size }}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Performance Metrics</h6>
                                <div class="row text-center">
                                    <div class="col-4">
                                        <div class="metric-value" style="font-size: 1.5rem; color: #007bff;">
                                            {{ "%.2f"|format(disease_model.training_accuracy * 100) }}%
                                        </div>
                                        <div class="metric-label" style="color: #6c757d;">Training</div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-value" style="font-size: 1.5rem; color: #28a745;">
                                            {{ "%.2f"|format(disease_model.validation_accuracy * 100) }}%
                                        </div>
                                        <div class="metric-label" style="color: #6c757d;">Validation</div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-value" style="font-size: 1.5rem; color: #ffc107;">
                                            {{ "%.2f"|format(disease_model.test_accuracy * 100) }}%
                                        </div>
                                        <div class="metric-label" style="color: #6c757d;">Test</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card model-card">
                    <div class="card-body text-center">
                        <h6>Model Performance Chart</h6>
                        <canvas id="diseaseModelChart" width="300" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Performance Comparison Chart -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card model-card">
                    <div class="card-body">
                        <h5 class="card-title">Model Accuracy Comparison</h5>
                        <canvas id="accuracyChart" width="800" height="400"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Disease Model Performance Chart
        const diseaseCtx = document.getElementById('diseaseModelChart').getContext('2d');
        new Chart(diseaseCtx, {
            type: 'doughnut',
            data: {
                labels: ['Training', 'Validation', 'Test'],
                datasets: [{
                    data: [
                        {{ disease_model.training_accuracy * 100 }},
                        {{ disease_model.validation_accuracy * 100 }},
                        {{ disease_model.test_accuracy * 100 }}
                    ],
                    backgroundColor: ['#007bff', '#28a745', '#ffc107'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // Accuracy Comparison Chart
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: [
                    {% for model in crop_models %}
                    '{{ model.name }}',
                    {% endfor %}
                    '{{ disease_model.name }}'
                ],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [
                        {% for model in crop_models %}
                        {{ model.accuracy * 100 }},
                        {% endfor %}
                        {{ disease_model.validation_accuracy * 100 }}
                    ],
                    backgroundColor: [
                        {% for model in crop_models %}
                        '{{ "#007bff" if model.status == "Deployed" else "#6c757d" }}',
                        {% endfor %}
                        '#dc3545'
                    ],
                    borderColor: [
                        {% for model in crop_models %}
                        '{{ "#0056b3" if model.status == "Deployed" else "#495057" }}',
                        {% endfor %}
                        '#c82333'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Models'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE, 
                                crop_models=MODEL_METRICS['crop_models'],
                                disease_model=MODEL_METRICS['disease_model'])

@app.route('/api/metrics')
def api_metrics():
    return json.dumps(MODEL_METRICS, indent=2)

if __name__ == '__main__':
    print("🚀 Starting FarmNex Model Metrics Dashboard...")
    print("📊 Dashboard available at: http://localhost:5002")
    print("🔗 API endpoint: http://localhost:5002/api/metrics")
    app.run(debug=True, port=5002)
