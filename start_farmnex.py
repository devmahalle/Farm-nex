#!/usr/bin/env python3
"""
FarmNex Startup Script
Handles Python 3.13 compatibility issues and starts both applications
"""

import warnings
import sys
import os

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def start_main_app():
    """Start the main FarmNex application"""
    print("🚀 Starting FarmNex Main Application...")
    print("📱 Available at: http://localhost:5001")
    print("=" * 50)
    
    # Change to app directory
    os.chdir(os.path.join(os.path.dirname(__file__), 'app'))
    
    # Import and run the application
    try:
        from application import app
        app.run(debug=False, port=5001, host='127.0.0.1')
    except Exception as e:
        print(f"❌ Error starting main app: {e}")
        return False
    return True

def start_metrics_dashboard():
    """Start the metrics dashboard"""
    print("📊 Starting FarmNex Metrics Dashboard...")
    print("📈 Available at: http://localhost:5002")
    print("=" * 50)
    
    # Change to app directory
    os.chdir(os.path.join(os.path.dirname(__file__), 'app'))
    
    # Import and run the dashboard
    try:
        from model_metrics_dashboard import app as dashboard_app
        dashboard_app.run(debug=False, port=5002, host='127.0.0.1')
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False
    return True

if __name__ == "__main__":
    print("🌱 FARMNEX AGRICULTURAL AI SYSTEM")
    print("=" * 50)
    print("Choose an option:")
    print("1. Start Main Application (Crop & Disease Detection)")
    print("2. Start Metrics Dashboard")
    print("3. Start Both Applications")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                start_main_app()
                break
            elif choice == '2':
                start_metrics_dashboard()
                break
            elif choice == '3':
                print("\n🔄 Starting both applications...")
                print("⚠️  Note: You'll need to run this script twice to start both apps")
                print("   Or use separate terminal windows for each application")
                start_main_app()
                break
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
