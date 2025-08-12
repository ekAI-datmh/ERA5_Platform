#!/usr/bin/env python3
"""
ERA5_Land Data Processing Module - Example Usage

This script demonstrates practical usage of the ERA5_Land processing module
with real-world examples for Vietnam climate data analysis.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the module to the path
sys.path.insert(0, str(Path(__file__).parent))

from era5_land_module import ERA5Downloader, DataPreprocessor, DataSearcher
from era5_land_module.config import *
from era5_land_module.utils import setup_logging


def example_1_download_vietnam_data():
    """
    Example 1: Download ERA5_Land data for Vietnam
    Download temperature and precipitation data for January 2024
    """
    print("\n=== Example 1: Download ERA5_Land Data for Vietnam ===")
    
    # Initialize downloader
    downloader = ERA5Downloader(
        dataset_type='hourly',
        format_type='grib',
        variables=['2m_temperature', 'total_precipitation', '2m_relative_humidity'],
        max_workers=2,  # Reduced for example
        delay_between_requests=2.0  # Increased delay for stability
    )
    
    # Download data for a short period (for demonstration)
    stats = downloader.download_data(
        start_date='2024-01-01',
        end_date='2024-01-03'  # Just 3 days for example
    )
    
    print(f"Download completed!")
    print(f"Successful downloads: {stats['successful_downloads']}")
    print(f"Failed downloads: {stats['failed_downloads']}")
    print(f"Total size: {stats['total_size']} bytes")
    
    return stats


def example_2_preprocess_data():
    """
    Example 2: Preprocess downloaded data
    Resample to 30m resolution and apply grid mask
    """
    print("\n=== Example 2: Preprocess Downloaded Data ===")
    
    # Check if we have downloaded data
    grib_dir = GRIB_DIR / "2024" / "01"
    if not grib_dir.exists():
        print("No downloaded data found. Please run Example 1 first.")
        return None
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_resolution='30m',
        interpolation_method='bilinear',
        fill_method='interpolate',
        mask_outside_grids=True,
        output_format='tif',
        compression='lzw',
        nodata_value=-9999,
        max_workers=2
    )
    
    # Preprocess data
    stats = preprocessor.preprocess_directory(
        input_dir=grib_dir,
        output_dir=PROCESSED_DIR,
        variables=['2m_temperature', 'total_precipitation']
    )
    
    print(f"Preprocessing completed!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Failed files: {stats['failed_files']}")
    
    return stats


def example_3_search_extract_data():
    """
    Example 3: Search and extract data for specific region
    Extract temperature data for Ho Chi Minh City area
    """
    print("\n=== Example 3: Search and Extract Data for Specific Region ===")
    
    # Define bounding box for Ho Chi Minh City area
    # (approximate coordinates)
    hcmc_bbox = ((106.5, 10.8), (106.8, 10.6))  # (top_left, bottom_right)
    
    # Initialize searcher
    searcher = DataSearcher(
        data_dir=DATA_DIR,
        max_bbox_size=1.0,  # Small area
        time_buffer=0,  # No buffer for exact dates
        output_resolution='30m',
        include_metadata=True,
        parallel_processing=True,
        max_workers=2
    )
    
    # Search and extract data
    stats = searcher.search_data(
        bbox=hcmc_bbox,
        start_date='2024-01-01',
        end_date='2024-01-03',
        variables=['2m_temperature'],
        output_dir=OUTPUT_DIR
    )
    
    print(f"Search completed!")
    print(f"Files found: {stats['total_files_found']}")
    print(f"Files extracted: {stats['files_extracted']}")
    print(f"Failed extractions: {stats['failed_extractions']}")
    
    return stats


def example_4_custom_region_analysis():
    """
    Example 4: Custom region analysis
    Analyze temperature patterns for Mekong Delta region
    """
    print("\n=== Example 4: Custom Region Analysis ===")
    
    # Define bounding box for Mekong Delta
    mekong_bbox = ((104.5, 10.5), (106.5, 8.5))  # (top_left, bottom_right)
    
    # Initialize searcher
    searcher = DataSearcher(
        data_dir=DATA_DIR,
        max_bbox_size=5.0,
        time_buffer=1,
        output_resolution='30m',
        include_metadata=True,
        parallel_processing=True,
        max_workers=2
    )
    
    # Search for multiple variables
    stats = searcher.search_data(
        bbox=mekong_bbox,
        start_date='2024-01-01',
        end_date='2024-01-03',
        variables=['2m_temperature', 'total_precipitation', '2m_relative_humidity'],
        output_dir=OUTPUT_DIR / "mekong_delta"
    )
    
    print(f"Mekong Delta analysis completed!")
    print(f"Files extracted: {stats['files_extracted']}")
    print(f"Total size: {stats['total_size']} bytes")
    
    return stats


def example_5_batch_processing():
    """
    Example 5: Batch processing for multiple months
    Download and process data for multiple months
    """
    print("\n=== Example 5: Batch Processing for Multiple Months ===")
    
    # Initialize downloader
    downloader = ERA5Downloader(
        dataset_type='hourly',
        format_type='grib',
        variables=['2m_temperature'],  # Single variable for efficiency
        max_workers=2,
        delay_between_requests=2.0
    )
    
    # Process multiple months
    months = [
        (2024, 1),
        (2024, 2),
        (2024, 3)
    ]
    
    total_stats = {
        'total_requests': 0,
        'successful_downloads': 0,
        'failed_downloads': 0,
        'total_size': 0
    }
    
    for year, month in months:
        print(f"\nProcessing {year}-{month:02d}...")
        
        try:
            stats = downloader.download_month(
                year=year,
                month=month,
                variables=['2m_temperature']
            )
            
            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
                
            print(f"Month {year}-{month:02d} completed: {stats['successful_downloads']} files")
            
        except Exception as e:
            print(f"Error processing {year}-{month:02d}: {e}")
    
    print(f"\nBatch processing completed!")
    print(f"Total successful downloads: {total_stats['successful_downloads']}")
    print(f"Total failed downloads: {total_stats['failed_downloads']}")
    print(f"Total size: {total_stats['total_size']} bytes")
    
    return total_stats


def example_6_data_validation():
    """
    Example 6: Data validation and quality checks
    Validate downloaded and processed data
    """
    print("\n=== Example 6: Data Validation and Quality Checks ===")
    
    # Initialize downloader for validation
    downloader = ERA5Downloader()
    
    # Validate downloaded data
    if GRIB_DIR.exists():
        print("Validating downloaded GRIB files...")
        validation_results = downloader.validate_downloads(GRIB_DIR)
        
        print(f"Total files: {validation_results['total_files']}")
        print(f"Valid files: {validation_results['valid_files']}")
        print(f"Invalid files: {validation_results['invalid_files']}")
        
        if validation_results['errors']:
            print("Errors found:")
            for error in validation_results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
    
    # Get download status
    if GRIB_DIR.exists():
        print("\nDownload status:")
        status = downloader.get_download_status(GRIB_DIR)
        print(f"Total files: {status['total_files']}")
        print(f"Total size: {status['total_size_formatted']}")
        print(f"Variables: {list(status['variables'].keys())}")
    
    return validation_results if 'validation_results' in locals() else None


def main():
    """Main function to run all examples."""
    print("ERA5_Land Data Processing Module - Example Usage")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging('ERA5_Land_Examples', 'INFO')
    
    # Check if CDS API key is available
    if not os.getenv('CDS_API_KEY'):
        print("Warning: CDS_API_KEY environment variable not set.")
        print("Please set your CDS API key before running examples:")
        print("export CDS_API_KEY='your-api-key-here'")
        print("\nContinuing with examples that don't require downloads...")
    
    try:
        # Run examples
        results = {}
        
        # Example 1: Download data (requires CDS API key)
        if os.getenv('CDS_API_KEY'):
            results['download'] = example_1_download_vietnam_data()
        else:
            print("\n=== Example 1: Download ERA5_Land Data ===")
            print("Skipped - CDS API key not available")
        
        # Example 2: Preprocess data
        results['preprocess'] = example_2_preprocess_data()
        
        # Example 3: Search and extract data
        results['search'] = example_3_search_extract_data()
        
        # Example 4: Custom region analysis
        results['custom_region'] = example_4_custom_region_analysis()
        
        # Example 5: Batch processing (requires CDS API key)
        if os.getenv('CDS_API_KEY'):
            results['batch'] = example_5_batch_processing()
        else:
            print("\n=== Example 5: Batch Processing ===")
            print("Skipped - CDS API key not available")
        
        # Example 6: Data validation
        results['validation'] = example_6_data_validation()
        
        # Summary
        print("\n" + "=" * 50)
        print("EXAMPLE EXECUTION SUMMARY")
        print("=" * 50)
        
        for example_name, result in results.items():
            if result:
                print(f"{example_name}: Completed successfully")
            else:
                print(f"{example_name}: Skipped or failed")
        
        print("\nAll examples completed!")
        print("Check the generated files in the data/ directory.")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        logger.error(f"Example execution error: {e}")


if __name__ == "__main__":
    main() 