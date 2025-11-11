import subprocess
import sys
import os

def update_requirements():
    """Update the requirements.txt file with current working versions"""
    
    print("Generating current requirements...")
    
    try:
        # Run pip freeze to get current versions
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                              capture_output=True, text=True, check=True)
        
        current_requirements = result.stdout
        
        # Write to requirements.txt
        with open('requirements.txt', 'w') as f:
            f.write(current_requirements)
        
        print("✅ requirements.txt updated successfully!")
        print(f"📦 Updated {len(current_requirements.strip().split())} packages")
        
        # Show some key packages
        key_packages = ['Flask', 'scikit-learn', 'numpy', 'pandas', 'torch', 'pillow']
        print("\n🔑 Key package versions:")
        for line in current_requirements.strip().split('\n'):
            for pkg in key_packages:
                if line.lower().startswith(pkg.lower()):
                    print(f"   {line}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pip freeze: {e}")
    except Exception as e:
        print(f"❌ Error updating requirements: {e}")

if __name__ == "__main__":
    update_requirements()

# Library	Old              new	
# Flask	        1.1.2	    3.1.1	
# scikit-learn	0.23.2	    1.7.1	
# numpy      	1.19.4	    2.3.2	
# pandas	    1.1.4	    2.3.1	
# torch	        1.7.0+cpu   2.8.0	
# torchvision	0.8.1+cpu   0.23.0	
# pillow	    8.0.1	    11.3.0
