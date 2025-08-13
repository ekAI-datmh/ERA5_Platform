#!/usr/bin/env python3
"""
Example ERA5_Land Data Download Script

This script downloads ERA5_Land data for all Vietnam grids for a single day and hour
as an example dataset for testing the preprocessing and search functionality.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the module to the path
sys.path.insert(0, str(Path(__file__).parent / 'era5_land_module'))

from era5_downloader import ERA5Downloader
from config import *
from utils import setup_logging


def main():
    """Download example ERA5_Land data for testing."""
    print("ERA5_Land Example Data Download")
    print("=" * 40)
    
    # Setup logging
    logger = setup_logging('ERA5_Example_Download', 'INFO')
    
    # Initialize downloader with minimal settings for testing
    downloader = ERA5Downloader(
        dataset_type='hourly',
        format_type='grib',
        variables=['2m_temperature', 'total_precipitation'],  # Just 2 variables for testing
        max_workers=2,  # Reduced for stability
        delay_between_requests=2.0  # Increased delay to be respectful to CDS
    )
    
    # Download data for January 1, 2024, 12:00 UTC only
    # This will download for all grids but only this specific time
    print("Downloading ERA5_Land data for all Vietnam grids...")
    print("Date: 2024-01-01")
    print("Time: 12:00 UTC")
    print("Variables: 2m_temperature, total_precipitation")
    print("Format: GRIB")
    print()
    
    try:
        # Download data
        stats = downloader.download_data(
            start_date='2024-01-01',
            end_date='2024-01-01'  # Just one day
        )
        
        print("\nDownload completed!")
        print(f"Successful downloads: {stats['successful_downloads']}")
        print(f"Failed downloads: {stats['failed_downloads']}")
        print(f"Total size: {stats['total_size']} bytes")
        
        if stats['successful_downloads'] > 0:
            print(f"\nData saved to: {GRIB_DIR}")
            print("You can now run the preprocessing and search examples!")
        else:
            print("\nNo data was downloaded. Please check the logs for errors.")
            
    except Exception as e:
        print(f"\nError during download: {e}")
        logger.error(f"Download error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nExample download completed successfully!")
    else:
        print("\nExample download failed!")
        sys.exit(1) 