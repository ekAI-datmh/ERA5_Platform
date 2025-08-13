"""
Utility functions for ERA5_Land data processing module.

This module contains helper functions used throughout the ERA5_Land processing pipeline.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import xarray as xr
import cdsapi

try:
    from .config import *
except ImportError:
    from config import *


def setup_logging(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Set up logging for a module.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(LOGGING_CONFIG['file'])
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_cds_api_key() -> Optional[str]:
    """
    Load CDS API key from environment or config file.
    
    Returns:
        CDS API key or None if not found
    """
    # Try environment variable first
    api_key = os.getenv('CDS_API_KEY')
    if api_key:
        return api_key
    
    # Try config file in current directory
    config_file = BASE_DIR / '.cdsapirc'
    if config_file.exists():
        with open(config_file, 'r') as f:
            for line in f:
                if line.startswith('key:'):
                    return line.split(':', 1)[1].strip()
    
    # Try config file in home directory
    home_config_file = Path.home() / '.cdsapirc'
    if home_config_file.exists():
        with open(home_config_file, 'r') as f:
            for line in f:
                if line.startswith('key:'):
                    return line.split(':', 1)[1].strip()
    
    return None


def create_cds_client() -> cdsapi.Client:
    """
    Create and configure CDS API client.
    
    Returns:
        Configured CDS API client
        
    Raises:
        ValueError: If CDS API key is not available
    """
    api_key = load_cds_api_key()
    if not api_key:
        raise ValueError("CDS API key not found. Please set CDS_API_KEY environment variable or create .cdsapirc file.")
    
    return cdsapi.Client(
        url=CDS_API_CONFIG['url'],
        key=api_key,
        timeout=CDS_API_CONFIG['timeout']
    )


def parse_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """
    Parse date range from string format.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    return start_dt, end_dt


def generate_date_list(start_date: datetime, end_date: datetime, 
                      frequency: str = 'hourly') -> List[datetime]:
    """
    Generate list of dates between start and end date.
    
    Args:
        start_date: Start datetime
        end_date: End datetime
        frequency: Frequency ('hourly', 'daily', 'monthly')
        
    Returns:
        List of datetime objects
    """
    dates = []
    current = start_date
    
    if frequency == 'hourly':
        delta = timedelta(hours=1)
    elif frequency == 'daily':
        delta = timedelta(days=1)
    elif frequency == 'monthly':
        delta = timedelta(days=30)  # Approximate
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    while current <= end_date:
        dates.append(current)
        current += delta
    
    return dates


def create_storage_paths(year: int, month: int, format_type: str = 'grib') -> Dict[str, Path]:
    """
    Create storage paths for data files.
    
    Args:
        year: Year
        month: Month
        format_type: Data format ('grib' or 'netcdf')
        
    Returns:
        Dictionary of paths
    """
    base_dir = GRIB_DIR if format_type == 'grib' else NETCDF_DIR
    year_dir = base_dir / str(year)
    month_dir = year_dir / f"{month:02d}"
    
    # Create directories
    year_dir.mkdir(exist_ok=True)
    month_dir.mkdir(exist_ok=True)
    
    return {
        'base': base_dir,
        'year': year_dir,
        'month': month_dir
    }


def load_grid_data() -> gpd.GeoDataFrame:
    """
    Load grid data from GeoJSON file.
    
    Returns:
        GeoDataFrame with grid information
    """
    if not GRID_CONFIG['geojson_path'].exists():
        raise FileNotFoundError(f"Grid file not found: {GRID_CONFIG['geojson_path']}")
    
    gdf = gpd.read_file(GRID_CONFIG['geojson_path'])
    
    # Parse dates field
    if GRID_CONFIG['dates_field'] in gdf.columns:
        gdf['parsed_dates'] = gdf[GRID_CONFIG['dates_field']].apply(
            lambda x: [datetime.strptime(d, '%Y-%m-%d') for d in x.split(';')] if pd.notna(x) else []
        )
    
    return gdf


def get_grid_bboxes() -> Dict[str, Dict[str, float]]:
    """
    Get bounding boxes for all grids.
    
    Returns:
        Dictionary mapping grid IDs to bounding boxes
    """
    gdf = load_grid_data()
    bboxes = {}
    
    for idx, row in gdf.iterrows():
        grid_id = row[GRID_CONFIG['grid_id_field']]
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        
        bboxes[grid_id] = {
            'west': bounds[0],
            'south': bounds[1],
            'east': bounds[2],
            'north': bounds[3]
        }
    
    return bboxes


def check_intersection(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
    """
    Check if two bounding boxes intersect.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        True if bounding boxes intersect
    """
    return not (bbox1['east'] < bbox2['west'] or 
                bbox1['west'] > bbox2['east'] or 
                bbox1['south'] > bbox2['north'] or 
                bbox1['north'] < bbox2['south'])


def ensure_cdo() -> bool:
    """
    Check if CDO (Climate Data Operators) is available.
    
    Returns:
        True if CDO is available
    """
    try:
        result = subprocess.run(['cdo', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def parse_resolution_to_degrees(resolution: str) -> float:
    """
    Parse resolution string to degrees.
    
    Args:
        resolution: Resolution string (e.g., '30m', '0.1deg')
        
    Returns:
        Resolution in degrees
    """
    if 'deg' in resolution:
        return float(resolution.replace('deg', ''))
    elif 'm' in resolution:
        meters = float(resolution.replace('m', ''))
        # Approximate conversion: 1 degree ≈ 111,000 meters at equator
        return meters / 111000.0
    else:
        raise ValueError(f"Unsupported resolution format: {resolution}")


def write_cdo_gridfile(lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                      resolution: float, output_file: Path) -> None:
    """
    Write CDO grid file for remapping.
    
    Args:
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        resolution: Resolution in degrees
        output_file: Output grid file path
    """
    nx = int((lon_max - lon_min) / resolution) + 1
    ny = int((lat_max - lat_min) / resolution) + 1
    
    with open(output_file, 'w') as f:
        f.write(f"gridtype = lonlat\n")
        f.write(f"xsize = {nx}\n")
        f.write(f"ysize = {ny}\n")
        f.write(f"xfirst = {lon_min}\n")
        f.write(f"xinc = {resolution}\n")
        f.write(f"yfirst = {lat_min}\n")
        f.write(f"yinc = {resolution}\n")


def cdo_remapbil_to_grid(input_file: Path, output_file: Path, 
                        grid_file: Path, overwrite: bool = False) -> bool:
    """
    Use CDO to remap data to a specific grid using bilinear interpolation.
    
    Args:
        input_file: Input file path
        output_file: Output file path
        grid_file: Grid file path
        overwrite: Whether to overwrite existing output file
        
    Returns:
        True if successful
    """
    if not ensure_cdo():
        raise RuntimeError("CDO not available")
    
    if output_file.exists() and not overwrite:
        return True
    
    try:
        cmd = ['cdo', '-s', '-O', 'remapbil', str(grid_file), str(input_file), str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def validate_file(file_path: Path, expected_size_min: int = 1000) -> bool:
    """
    Validate downloaded file.
    
    Args:
        file_path: File path to validate
        expected_size_min: Minimum expected file size in bytes
        
    Returns:
        True if file is valid
    """
    if not file_path.exists():
        return False
    
    if file_path.stat().st_size < expected_size_min:
        return False
    
    return True


def cleanup_temp_files(temp_dir: Path) -> None:
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Temporary directory path
    """
    if temp_dir.exists():
        for file_path in temp_dir.glob('*'):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.warning(f"Failed to cleanup {file_path}: {e}")


def save_metadata(metadata: Dict, output_path: Path) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(metadata_path: Path) -> Dict:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Metadata file path
        
    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def calculate_bbox_area(bbox: Dict[str, float]) -> float:
    """
    Calculate area of bounding box in square degrees.
    
    Args:
        bbox: Bounding box dictionary
        
    Returns:
        Area in square degrees
    """
    width = bbox['east'] - bbox['west']
    height = bbox['north'] - bbox['south']
    return width * height


def estimate_download_size(bbox: Dict[str, float], variables: List[str], 
                          start_date: datetime, end_date: datetime,
                          format_type: str = 'grib') -> int:
    """
    Estimate download size in bytes.
    
    Args:
        bbox: Bounding box
        variables: List of variables
        start_date: Start date
        end_date: End date
        format_type: Data format
        
    Returns:
        Estimated size in bytes
    """
    # Rough estimation: 1 variable = ~1MB per day for ERA5_Land
    days = (end_date - start_date).days + 1
    base_size_per_day = 1024 * 1024  # 1MB
    
    total_size = len(variables) * days * base_size_per_day
    
    if format_type == 'netcdf':
        total_size *= 0.8  # NetCDF is typically smaller
    
    return int(total_size)


def create_progress_callback(total_items: int, description: str = "Processing"):
    """
    Create a progress callback function.
    
    Args:
        total_items: Total number of items to process
        description: Description for progress bar
        
    Returns:
        Progress callback function
    """
    from tqdm import tqdm
    
    pbar = tqdm(total=total_items, desc=description, unit='items')
    
    def callback(completed_items: int = 1):
        pbar.update(completed_items)
        if pbar.n >= total_items:
            pbar.close()
    
    return callback 