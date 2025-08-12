#!/usr/bin/env python3
"""Model Manager - Lazy downloading and management of AI models"""
import urllib.request
import tarfile
import shutil
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import importlib.metadata

def get_package_version() -> str:
    """Get the current package version"""
    try:
        return importlib.metadata.version("bracketbot-ai")
    except importlib.metadata.PackageNotFoundError:
        # Fallback to reading from pyproject.toml or __init__.py
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "0.0.1"  # Default fallback

MODELS_DIR = Path(__file__).parent / "models"

# Model configurations
MODELS = [
    "yolo11s",
    "SenseVoiceSmall"
]


def fetch_model(url: str, destination: Path) -> bool:
    """Download a file using wget"""
    try:
        print(f"📥 Downloading: {url}")
        print(f"   Destination: {destination}")
        
        # Create parent directories
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print(f"✅ Downloaded: {destination.name}")
            return True
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract tar.gz archive"""
    try:
        print(f"📦 Extracting: {archive_path.name}")
        tar = tarfile.open(archive_path)
        tar.extractall(extract_to)
        tar.close()
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return False
    return True

def ensure_model(model_name: str) -> Path:
    """Ensure model files are available, download if missing"""
    
    # Check if model is in our known models
    if model_name not in MODELS:
        print(f"⚠️  Unknown audio model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return []
    

    model_dir = MODELS_DIR / model_name if (MODELS_DIR / model_name).is_dir() else MODELS_DIR
    model_file = model_dir / f"{model_name}.rknn"

    if not model_file.exists():
        print(f"📦 Audio model '{model_name}' not found, downloading...")
        
        # Download model archive
        version = get_package_version()
        download_url = f"https://github.com/Bracket-Bot-Inc/bracketbot-ai/releases/download/v{version}/{model_name}.tar.gz"
        
        temp_archive = MODELS_DIR / f"temp_{model_name}.tar.gz"
        
        try:
            if not fetch_model(download_url, temp_archive):
                return []
            if not extract_archive(temp_archive, MODELS_DIR):
                return []
            print(f"✅ Audio model '{model_name}' successfully installed")
            
        except Exception as e:
            print(f"❌ Error installing audio model '{model_name}': {e}")
            return []
        finally:
            if temp_archive.exists():
                temp_archive.unlink()

    model_dir = MODELS_DIR / model_name if (MODELS_DIR / model_name).is_dir() else MODELS_DIR
    return model_dir, model_file

def list_available_models() -> Dict[str, List[str]]:
    """List all available models that can be downloaded"""
    return MODELS.keys()