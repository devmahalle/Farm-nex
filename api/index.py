"""
Vercel serverless function entry point for FarmNex Flask application
"""
import sys
import os

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
sys.path.insert(0, app_dir)

# Change to app directory for relative paths to work
os.chdir(app_dir)

# Import the Flask app
from application import app

# Vercel expects the app to be exported
# The Flask app will be automatically detected by Vercel's Python runtime

