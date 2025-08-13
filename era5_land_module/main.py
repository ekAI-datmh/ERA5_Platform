#!/usr/bin/env python3
"""
ERA5_Land Data Processing Module - Main Script

This script demonstrates how to use the ERA5_Land processing module for:
1. Downloading ERA5_Land data for Vietnam
2. Preprocessing downloaded data (resampling, masking, etc.)
3. Searching and extracting data for specific regions and time periods

Usage:
    python main.py --task download --start-date 2024-01-01 --end-date 2024-01-31
    python main.py --task preprocess --input-dir data/era5_vietnam/grib
    python main.py --task search --bbox 105,16,110,10 --start-date 2024-01-01 --end-date 2024-01-31 --variables 2m_temperature
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add the module to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from era5_land_module import ERA5Downloader, DataPreprocessor, DataSearcher
    from era5_land_module.config import *
    from era5_land_module.utils import setup_logging
except ImportError:
    from era5_downloader import ERA5Downloader
    from data_preprocessor import DataPreprocessor
    from data_searcher import DataSearcher
    from config import *
    from utils import setup_logging


def main():
    """Main function to run ERA5_Land processing tasks."""
    parser = argparse.ArgumentParser(
        description="ERA5_Land Data Processing Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download ERA5_Land data for Vietnam (January 2024)
  python main.py --task download --start-date 2024-01-01 --end-date 2024-01-31 --format grib

  # Preprocess downloaded data (resample to 30m, apply grid mask)
  python main.py --task preprocess --input-dir data/era5_vietnam/grib --resolution 30m

  # Search and extract data for specific region
  python main.py --task search --bbox 105,16,110,10 --start-date 2024-01-01 --end-date 2024-01-31 --variables 2m_temperature
        """
    )
    
    # Common arguments
    parser.add_argument('--task', required=True, choices=['download', 'preprocess', 'search'],
                       help='Task to perform')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Download task arguments
    parser.add_argument('--start-date', type=str,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--format', choices=['grib', 'netcdf'], default='grib',
                       help='Output format for downloads')
    parser.add_argument('--variables', nargs='+',
                       help='Variables to download (if not specified, uses default list)')
    parser.add_argument('--dataset-type', choices=['hourly', 'daily'], default='hourly',
                       help='Dataset type (hourly or daily)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for downloads')
    
    # Preprocessing task arguments
    parser.add_argument('--input-dir', type=str,
                       help='Input directory containing ERA5_Land files')
    parser.add_argument('--resolution', type=str, default='30m',
                       help='Target resolution for resampling (e.g., 30m, 0.1deg)')
    parser.add_argument('--interpolation', choices=['bilinear', 'nearest', 'cubic'], default='bilinear',
                       help='Interpolation method for resampling')
    parser.add_argument('--fill-method', choices=['interpolate', 'fill', 'none'], default='interpolate',
                       help='Method for filling NaN values')
    parser.add_argument('--mask-grids', action='store_true', default=True,
                       help='Mask areas outside defined grids')
    parser.add_argument('--output-format', choices=['tif', 'grib', 'netcdf'], default='tif',
                       help='Output format for processed files')
    
    # Search task arguments
    parser.add_argument('--bbox', type=str,
                       help='Bounding box as "west,south,east,north" (e.g., "105,16,110,10")')
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing ERA5_Land data')
    parser.add_argument('--max-bbox-size', type=float, default=5.0,
                       help='Maximum bounding box size in degrees')
    parser.add_argument('--time-buffer', type=int, default=1,
                       help='Buffer in days around requested time period')
    parser.add_argument('--output-resolution', type=str, default='30m',
                       help='Output resolution for search results')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging('ERA5_Land_Main', log_level)
    
    logger.info(f"Starting ERA5_Land processing task: {args.task}")
    
    try:
        if args.task == 'download':
            run_download_task(args, logger)
        elif args.task == 'preprocess':
            run_preprocess_task(args, logger)
        elif args.task == 'search':
            run_search_task(args, logger)
        else:
            logger.error(f"Unknown task: {args.task}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during {args.task} task: {e}")
        sys.exit(1)
    
    logger.info("Task completed successfully!")


def run_download_task(args, logger):
    """Run the download task."""
    logger.info("Starting ERA5_Land download task")
    
    # Validate arguments
    if not args.start_date or not args.end_date:
        logger.error("Start date and end date are required for download task")
        sys.exit(1)
    
    # Create output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader
    downloader = ERA5Downloader(
        dataset_type=args.dataset_type,
        format_type=args.format,
        variables=args.variables,
        max_workers=4,
        retry_attempts=3,
        delay_between_requests=1.0
    )
    
    # Download data
    logger.info(f"Downloading ERA5_Land data from {args.start_date} to {args.end_date}")
    logger.info(f"Format: {args.format}, Dataset type: {args.dataset_type}")
    
    stats = downloader.download_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=output_dir
    )
    
    # Print results
    logger.info("Download completed!")
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Successful downloads: {stats['successful_downloads']}")
    logger.info(f"Failed downloads: {stats['failed_downloads']}")
    logger.info(f"Total size: {stats['total_size']} bytes")
    
    if output_dir:
        # Get download status
        status = downloader.get_download_status(output_dir)
        logger.info(f"Files downloaded: {status['total_files']}")
        logger.info(f"Total size: {status['total_size_formatted']}")


def run_preprocess_task(args, logger):
    """Run the preprocessing task."""
    logger.info("Starting ERA5_Land preprocessing task")
    
    # Validate arguments
    if not args.input_dir:
        logger.error("Input directory is required for preprocessing task")
        sys.exit(1)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_resolution=args.resolution,
        interpolation_method=args.interpolation,
        fill_method=args.fill_method,
        mask_outside_grids=args.mask_grids,
        output_format=args.output_format,
        compression='lzw',
        nodata_value=-9999,
        max_workers=4
    )
    
    # Preprocess data
    logger.info(f"Preprocessing data in {input_dir}")
    logger.info(f"Target resolution: {args.resolution}")
    logger.info(f"Output format: {args.output_format}")
    
    stats = preprocessor.preprocess_directory(
        input_dir=input_dir,
        output_dir=PROCESSED_DIR,
        variables=args.variables
    )
    
    # Print results
    logger.info("Preprocessing completed!")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Processed files: {stats['processed_files']}")
    logger.info(f"Failed files: {stats['failed_files']}")
    logger.info(f"Total size: {stats['total_size']} bytes")


def run_search_task(args, logger):
    """Run the search task."""
    logger.info("Starting ERA5_Land search task")
    
    # Validate arguments
    if not args.bbox:
        logger.error("Bounding box is required for search task")
        sys.exit(1)
    
    if not args.start_date or not args.end_date:
        logger.error("Start date and end date are required for search task")
        sys.exit(1)
    
    # Parse bounding box
    try:
        bbox_coords = [float(x) for x in args.bbox.split(',')]
        if len(bbox_coords) != 4:
            raise ValueError("Bounding box must have 4 coordinates")
        
        bbox = ((bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]))
        logger.info(f"Searching in bounding box: {bbox}")
        
    except Exception as e:
        logger.error(f"Invalid bounding box format: {args.bbox}. Expected: west,south,east,north")
        sys.exit(1)
    
    # Set data directory
    data_dir = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Use default data directory
        data_dir = DATA_DIR
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Initialize searcher
    searcher = DataSearcher(
        data_dir=data_dir,
        max_bbox_size=args.max_bbox_size,
        time_buffer=args.time_buffer,
        output_resolution=args.output_resolution,
        include_metadata=True,
        parallel_processing=True,
        max_workers=4
    )
    
    # Search and extract data
    logger.info(f"Searching data from {args.start_date} to {args.end_date}")
    logger.info(f"Variables: {args.variables if args.variables else 'all'}")
    
    stats = searcher.search_data(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        variables=args.variables,
        output_dir=OUTPUT_DIR
    )
    
    # Print results
    logger.info("Search completed!")
    logger.info(f"Files found: {stats['total_files_found']}")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Files extracted: {stats['files_extracted']}")
    logger.info(f"Failed extractions: {stats['failed_extractions']}")
    logger.info(f"Total size: {stats['total_size']} bytes")
    
    # Get search status
    status = searcher.get_search_status(OUTPUT_DIR)
    logger.info(f"Output files: {status['total_files']}")
    logger.info(f"Total size: {status['total_size_formatted']}")


if __name__ == "__main__":
    main() 