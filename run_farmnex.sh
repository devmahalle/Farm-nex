#!/bin/bash

# FarmNex Startup Script
echo "🌱 Starting FarmNex Agricultural AI System..."

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Suppress warnings and start the main application
echo "🚀 Starting Main Application on http://localhost:5001"
cd app
python -c "
import warnings
warnings.filterwarnings('ignore')
exec(open('application.py').read())
" &

# Wait a moment
sleep 2

# Start metrics dashboard in background
echo "📊 Starting Metrics Dashboard on http://localhost:5002"
python -c "
import warnings
warnings.filterwarnings('ignore')
exec(open('model_metrics_dashboard.py').read())
" &

echo ""
echo "✅ FarmNex is now running!"
echo "🌱 Main Application: http://localhost:5001"
echo "📊 Metrics Dashboard: http://localhost:5002"
echo ""
echo "Press Ctrl+C to stop all applications"

# Wait for user to stop
wait
