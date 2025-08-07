#!/usr/bin/env python3
"""
Generic ONNX to RKNN Model Converter
Downloads ONNX models from URLs and converts to RKNN format for RK3588 NPU
"""

import argparse
import sys
import subprocess
import urllib.parse
from pathlib import Path
from rknn.api import RKNN

def download_onnx_model(url, temp_dir="temp"):
    """
    Download ONNX model from URL using wget
    
    Args:
        url: URL to ONNX model
        temp_dir: Temporary directory for download
    
    Returns:
        Path to downloaded ONNX file or None if failed
    """
    print(f"📥 Downloading ONNX model from: {url}")
    
    # Create temp directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    
    # Extract filename from URL
    parsed_url = urllib.parse.urlparse(url)
    filename = Path(parsed_url.path).name
    if not filename.endswith('.onnx'):
        print("❌ Error: URL must point to an .onnx file")
        return None
    
    output_path = temp_path / filename
    
    try:
        # Use wget to download the file
        cmd = ['wget', '-O', str(output_path), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return None
        
        print(f"✅ Downloaded: {output_path}")
        return output_path
        
    except FileNotFoundError:
        print("❌ Error: wget not found. Please install wget.")
        return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def convert_onnx_to_rknn(onnx_path, do_quantization=True, dataset_path=None):
    """
    Convert ONNX model to RKNN format
    
    Args:
        onnx_path: Path to ONNX model
        do_quantization: Enable INT8 quantization for better NPU performance
        dataset_path: Path to dataset file for quantization
    
    Returns:
        Path to output RKNN file or None if failed
    """
    print(f"🔄 Converting {onnx_path} to RKNN format...")
    print("=" * 60)
    
    # Validate input
    if not Path(onnx_path).exists():
        print(f"❌ Error: ONNX model not found: {onnx_path}")
        return None
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate output filename in models directory
    model_name = Path(onnx_path).stem
    output_path = models_dir / f"{model_name}_rk3588.rknn"
    
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    # Configure for RK3588
    print("📋 Configuring for RK3588 NPU...")
    ret = rknn.config(
        mean_values=[0, 0, 0],          # Common for most models
        std_values=[255, 255, 255],      # Normalize to 0-1
        target_platform='rk3588',
        optimization_level=3,            # Maximum optimization
        quantized_algorithm='normal',
        quantized_method='channel',
        quant_img_RGB2BGR=False,         # Keep RGB order
        float_dtype='float16'            # FP16 for better accuracy
    )
    if ret != 0:
        print('❌ Config failed!')
        return None
    
    # Load ONNX model
    print(f"\n📥 Loading ONNX model: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print('❌ Load ONNX failed!')
        print('   Make sure the ONNX model is valid and compatible')
        return None
    
    # Build the model
    print("\n🔨 Building RKNN model...")
    if do_quantization:
        # Create dataset file if not provided
        if dataset_path is None:
            dataset_path = "dataset.txt"
            if not Path(dataset_path).exists():
                print("📝 Creating dataset file for quantization...")
                with open(dataset_path, "w") as f:
                    # Add test image(s) for calibration
                    if Path("test.jpg").exists():
                        f.write("test.jpg\n")
                    else:
                        print("⚠️  Warning: No test.jpg found for quantization calibration")
                        print("   Using default quantization parameters")
                        dataset_path = None
        
        if dataset_path:
            print(f"   Using dataset: {dataset_path}")
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            ret = rknn.build(do_quantization=False)
    else:
        print("   Building without quantization (FP16)")
        ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        print('❌ Build failed!')
        return None
    
    # Export RKNN model
    print(f"\n💾 Exporting RKNN model to: {output_path}")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        print('❌ Export failed!')
        return None
    
    print(f"\n✅ Conversion successful!")
    print(f"   Output file: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show model info
    print("\n📊 Model Information:")
    print("   Platform: RK3588")
    print("   Optimization: Level 3")
    print(f"   Quantization: {'INT8' if do_quantization else 'FP16'}")
    print("   NPU Cores: 3 (6 TOPS)")
    
    # Release resources
    rknn.release()
    return output_path

def cleanup_temp_files(onnx_path):
    """Clean up temporary ONNX file"""
    try:
        if Path(onnx_path).exists():
            Path(onnx_path).unlink()
            print(f"🗑️  Cleaned up temporary file: {onnx_path}")
        
        # Also clean up temp directory if empty
        temp_dir = Path(onnx_path).parent
        if temp_dir.name == "temp" and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            print(f"🗑️  Removed empty temp directory")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up {onnx_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download ONNX models from URLs and convert to RKNN format')
    parser.add_argument('--url', '-u', required=True, help='URL to ONNX model')
    parser.add_argument('--no-quant', action='store_true', help='Disable INT8 quantization')
    parser.add_argument('--dataset', '-d', help='Dataset file for quantization calibration')
    
    args = parser.parse_args()
    
    print("🚀 ONNX to RKNN Converter for RK3588")
    print("=" * 60)
    
    # Download the ONNX model
    onnx_path = download_onnx_model(args.url)
    if not onnx_path:
        print("\n❌ Download failed. Please check the URL and try again.")
        sys.exit(1)
    
    try:
        # Convert the model
        output_path = convert_onnx_to_rknn(
            onnx_path,
            not args.no_quant,
            args.dataset
        )
        
        if output_path:
            print("\n🎉 Model ready for NPU inference!")
            print(f"   Saved to: {output_path}")
            print("\n💡 Tips:")
            print("   - Use rknnlite.api.RKNNLite for inference")
            print("   - Distribute models across NPU cores for best performance")
            print("   - INT8 models are 2-4x faster than FP16")
        else:
            print("\n❌ Conversion failed. Please check the error messages above.")
            sys.exit(1)
    
    finally:
        # Always clean up temporary files
        cleanup_temp_files(onnx_path)

if __name__ == "__main__":
    main() 