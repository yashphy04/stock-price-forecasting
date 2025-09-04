import subprocess
import sys

# List of required packages
required = [
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "xgboost",
    "joblib",
    "jupyter"
]

def install(package):
    """Install package via pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")

def main():
    print("🔍 Checking required packages...\n")
    for pkg in required:
        try:
            __import__(pkg if pkg != "scikit-learn" else "sklearn")
            print(f"✅ {pkg} already installed")
        except ImportError:
            print(f"⬇ Installing {pkg}...")
            install(pkg)
    print("\n🎉 All required packages are ready!")

if __name__ == "__main__":
    main()
