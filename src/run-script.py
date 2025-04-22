#!/usr/bin/env python3
"""
Launcher script for the Text Summarization Pipeline App.
This script ensures proper setup before launching the Streamlit application.
"""

import os
import sys
import subprocess
import platform
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import lancedb
        import ollama
        from rich.console import Console
        print("✅ All required Python packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False

def check_ollama():
    """Check if Ollama is available and the required model is installed."""
    try:
        import ollama
        models = ollama.list()
        llama_models = [m['name'] for m in models['models'] if 'llama3.2' in m['name']]
        
        if not llama_models:
            print("⚠️ Llama3.2 model not found in Ollama.")
            print("You may need to install it with: ollama pull llama3.2")
            return False
        
        print(f"✅ Found Llama models: {', '.join(llama_models)}")
        return True
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        print("Make sure Ollama is installed and running.")
        return False

def ensure_directories():
    """Ensure required directories exist."""
    # Create LanceDB directory if it doesn't exist
    Path("./lancedb").mkdir(exist_ok=True)
    print("✅ Directories setup complete.")
    return True

def launch_app():
    """Launch the Streamlit application."""
    print("🚀 Launching Streamlit application...")
    
    # Determine the path to the app.py file
    app_path = Path("src/app.py")
    
    # Run Streamlit
    try:
        # Open browser automatically
        webbrowser.open('http://localhost:8501', new=2)
        
        # Start Streamlit
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Text Summarization Pipeline - Setup Check")
    print("=" * 60)
    
    # Run checks
    deps_ok = check_dependencies()
    ollama_ok = check_ollama()
    dirs_ok = ensure_directories()
    
    if deps_ok and ollama_ok and dirs_ok:
        print("\n✅ All checks passed! Ready to launch application.")
        launch_app()
    else:
        print("\n❌ Some checks failed. Please resolve the issues before running the app.")
        sys.exit(1)
