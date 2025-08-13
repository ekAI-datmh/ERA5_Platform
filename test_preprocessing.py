#!/usr/bin/env python3
"""
Test Preprocessing with Downloaded ERA5_Land Data

This script tests the preprocessing functionality with the actual downloaded data.
"""

import os
import sys
from pathlib import Path

# Add the module to the path
sys.path.insert(0, str(Path(__file__).parent / 'era5_land_module'))

from data_preprocessor import DataPreprocessor
from config import *
from utils import setup_logging


def main():
    """Test preprocessing with downloaded data."""
    print("Testing ERA5_Land Data Preprocessing")
    print("=" * 40)
    
    # Setup logging
    logger = setup_logging('Test_Preprocessing', 'INFO')
    
    # Check if we have downloaded data
    grib_dir = GRIB_DIR / "2024" / "01"
    if not grib_dir.exists():
        print("No downloaded data found. Please run the download script first.")
        return False
    
    # List the downloaded files
    print(f"Found downloaded files in: {grib_dir}")
    grib_files = list(grib_dir.glob("*.grib"))
    print(f"Total GRIB files: {len(grib_files)}")
    
    # Show the files we downloaded
    our_files = [f for f in grib_files if f.name.startswith(('2m_temperature', 'total_precipitation'))]
    print(f"Our downloaded files: {len(our_files)}")
    for f in our_files:
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_resolution='30m',
        interpolation_method='bilinear',
        fill_method='interpolate',
        mask_outside_grids=True,
        output_format='tif',
        compression='lzw',
        nodata_value=-9999,
        max_workers=1  # Single worker for testing
    )
    
    # Test preprocessing on one file
    if our_files:
        test_file = our_files[0]
        print(f"\nTesting preprocessing on: {test_file.name}")
        
        try:
            # Create output directory
            output_dir = PROCESSED_DIR / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Preprocess the file
            result = preprocessor.preprocess_file(
                input_file=test_file
            )
            
            if result and result.get('success'):
                print(f"Preprocessing successful!")
                print(f"Output file: {result.get('output_file', 'Unknown')}")
                print(f"File size: {result.get('file_size', 0)} bytes")
                
                # Check if output file exists
                output_file = result.get('output_file')
                if output_file and Path(output_file).exists():
                    print(f"Output file size: {Path(output_file).stat().st_size} bytes")
                else:
                    print("Warning: Output file not found")
            else:
                print("Preprocessing failed")
                if result:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            logger.error(f"Preprocessing error: {e}")
            return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nPreprocessing test completed!")
    else:
        print("\nPreprocessing test failed!")
        sys.exit(1) 