# FarmNex - Agricultural Assistant

## Project Overview
FarmNex is a comprehensive agricultural assistant web application designed to help farmers make informed decisions about crop selection, disease management, and fertilizer usage. The application leverages machine learning models to provide personalized recommendations based on various input parameters.

## Features
The application offers three main services:

1. **Crop Recommendation**
   - Recommends the most suitable crop based on soil and climate parameters
   - Input parameters: Nitrogen (N), Phosphorus (P), Potassium (K), pH, rainfall, temperature, and humidity
   - Uses a Random Forest classifier model for predictions

2. **Plant Disease Detection**
   - Identifies diseases in plants from uploaded images
   - Supports multiple crops including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
   - Uses a ResNet9 deep learning model for image classification
   - Can detect 38 different classes of plant diseases

3. **Fertilizer Recommendation**
   - Suggests appropriate fertilizers based on soil nutrient content and crop type
   - Input parameters: Crop type, Nitrogen (N), Phosphorus (P), and Potassium (K)
   - Provides detailed recommendations for addressing nutrient deficiencies

## Project Structure
```
FarmNex/
├── .venv/                      # Virtual environment
├── Data-processed/             # Processed datasets
│   ├── crop_recommendation.csv
│   └── fertilizer.csv
├── Data-raw/                   # Raw datasets
│   ├── Fertilizer.csv
│   ├── FertilizerData.csv
│   ├── MergeFileCrop.csv
│   ├── cpdata.csv
│   └── raw_districtwise_yield_data.csv
├── app/                        # Main application directory
│   ├── .ebextensions/          # Elastic Beanstalk configuration
│   ├── Data/                   # Application data files
│   ├── models/                 # Trained ML models
│   │   ├── RandomForest_new.pkl
│   │   └── plant_disease_model.pth
│   ├── static/                 # Static files (CSS, JS, images)
│   ├── templates/              # HTML templates
│   ├── utils/                  # Utility modules
│   │   ├── disease.py          # Disease dictionary and functions
│   │   ├── fertilizer.py       # Fertilizer dictionary and functions
│   │   └── model.py            # ML model definitions
│   ├── application.py          # Main Flask application
│   ├── config.py               # Configuration settings
│   └── requirements.txt        # Application dependencies
├── models/                     # Additional trained models
├── notebooks/                  # Jupyter notebooks for model training
└── requirements.txt            # Project dependencies
```

## Backend Process Flow

### 1. Crop Recommendation System
- **Data Collection**: Uses soil parameters (N, P, K, pH) and climate data (rainfall, temperature, humidity)
- **Model**: Random Forest classifier trained on crop_recommendation.csv dataset
- **Process Flow**:
  1. User inputs soil and climate parameters via web form
  2. Flask backend receives the data and passes it to the model
  3. Model predicts the most suitable crop
  4. Result is displayed to the user

### 2. Plant Disease Detection System
- **Data Processing**: Uses image processing techniques to analyze plant images
- **Model**: ResNet9 CNN architecture trained on plant disease dataset
- **Process Flow**:
  1. User uploads an image of a plant leaf
  2. Image is preprocessed (resized, normalized)
  3. Processed image is fed to the ResNet9 model
  4. Model classifies the image into one of 38 disease categories
  5. Disease information and treatment suggestions are displayed

### 3. Fertilizer Recommendation System
- **Data Analysis**: Compares soil nutrient levels with crop requirements
- **Process Flow**:
  1. User selects crop type and inputs current N, P, K values
  2. System calculates the nutrient deficiency/excess
  3. Appropriate fertilizer recommendations are provided based on the analysis

## Setup and Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Installation Steps

1. **Clone or download the project**
   ```
   git clone <repository-url>
   ```
   or download and extract the ZIP file

2. **Navigate to the project directory**
   ```
   cd FarmNex
   ```

3. **Create and activate a virtual environment**
   - Windows:
     ```
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. **Install required dependencies**
   ```
   pip install -r app/requirements.txt
   ```
   
   Note: Some key dependencies include:
   - Flask (web framework)
   - PyTorch (for deep learning models)
   - scikit-learn (for machine learning models)
   - pandas (for data manipulation)
   - numpy (for numerical operations)
   - Pillow (for image processing)

5. **Navigate to the app directory**
   ```
   cd app
   ```

6. **Run the application**
   ```
   python application.py
   ```

7. **Access the application**
   Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Missing dependencies**
   - Error: `ModuleNotFoundError: No module named 'torch'`
   - Solution: Install the missing package using pip
     ```
     pip install torch torchvision
     ```

2. **Model loading errors**
   - Error: `FileNotFoundError: [Errno 2] No such file or directory: 'models/plant_disease_model.pth'`
   - Solution: Ensure all model files are in the correct directory (app/models/)

3. **Version compatibility issues**
   - Error: `InconsistentVersionWarning: Trying to unpickle estimator...`
   - Solution: This warning indicates the model was trained with a different version of scikit-learn. The model should still work, but consider retraining if issues occur.

## Data Sources and Model Training

The machine learning models were trained using:
1. Crop recommendation dataset - Contains soil parameters and suitable crops
2. Plant disease images - Dataset of plant leaves with various diseases
3. Fertilizer recommendation data - Information about crop nutrient requirements

The Jupyter notebooks in the `notebooks/` directory contain the code used for data preprocessing, model training, and evaluation.

## Future Enhancements
- Weather API integration for real-time climate data
- Mobile application development
- Support for more regional crops and diseases
- Multilingual support for different regions
- Yield prediction functionality

---

## Deployment

### Deploying to Vercel

1. **Install Vercel CLI** (optional, for local testing):
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**:
   - Option A: Using Vercel Dashboard
     1. Go to [vercel.com](https://vercel.com)
     2. Import your GitHub repository
     3. Vercel will automatically detect the Python project
     4. Deploy!

   - Option B: Using Vercel CLI
     ```bash
     vercel
     ```

3. **Important Notes for Vercel Deployment**:
   - The project is configured with `vercel.json` and `api/index.py` for serverless deployment
   - Model files must be included in the repository (they're in `app/models/`)
   - Static files are served from `app/static/`
   - The application runs on Vercel's serverless Python runtime

### GitHub Repository

The project is configured for GitHub deployment. To push to GitHub:

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/devmahalle/Farm-nex.git
git push -u origin main
```

---

For any questions or issues, please contact the project maintainers.