"""
Configuration file for ERA5_Land data processing module.

This file contains all the configuration parameters, constants, and settings
used throughout the ERA5_Land processing pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "era5_vietnam"

# Data storage schema
GRIB_DIR = DATA_DIR / "grib"
NETCDF_DIR = DATA_DIR / "netcdf"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Create directories if they don't exist
for dir_path in [GRIB_DIR, NETCDF_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ERA5_Land dataset configuration
ERA5_LAND_DATASET = "reanalysis-era5-single-levels"  # Try standard ERA5 first
ERA5_LAND_HOURLY_DATASET = "reanalysis-era5-single-levels"

# Vietnam bounding box (approximate)
VIETNAM_BBOX = {
    'north': 23.5,  # Northernmost latitude
    'south': 8.5,   # Southernmost latitude
    'west': 102.0,  # Westernmost longitude
    'east': 110.0   # Easternmost longitude
}

# ERA5_Land variables for hourly data
ERA5_LAND_HOURLY_VARIABLES = [
    '2m_temperature',
    '2m_relative_humidity',
    'surface_pressure',
    'total_precipitation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards_hourly',
    'surface_thermal_radiation_downwards_hourly',
    'mean_sea_level_pressure',
    'evaporation',
    'runoff',
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'soil_temperature_level_4',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
]

# ERA5_Land variables for daily data
ERA5_LAND_DAILY_VARIABLES = [
    '2m_temperature_max',
    '2m_temperature_min',
    'total_precipitation',
    'surface_pressure',
    '2m_relative_humidity',
    '10m_wind_speed',
    'surface_solar_radiation_downwards_daily',
    'surface_thermal_radiation_downwards_daily',
    'evaporation',
    'runoff'
]

# CDS API configuration
CDS_API_CONFIG = {
    'url': 'https://cds.climate.copernicus.eu/api',  # Use the URL from .cdsapirc
    'key': None,  # Will be loaded from environment or config file
    'timeout': 3600,  # 1 hour timeout
    'retry_attempts': 3,
    'retry_delay': 60  # seconds
}

# Processing configuration
PROCESSING_CONFIG = {
    'target_resolution': '30m',  # Target resolution for resampling
    'interpolation_method': 'bilinear',  # Interpolation method for resampling
    'fill_method': 'interpolate',  # Method for filling NaN values
    'mask_outside_grids': True,  # Whether to mask areas outside defined grids
    'output_format': 'tif',  # Output format for processed data
    'compression': 'lzw',  # Compression method for output files
    'nodata_value': -9999  # NoData value for output files
}

# Search configuration
SEARCH_CONFIG = {
    'max_bbox_size': 5.0,  # Maximum bounding box size in degrees
    'time_buffer': 1,  # Buffer in days around requested time period
    'output_resolution': '30m',  # Output resolution for search results
    'include_metadata': True,  # Whether to include metadata in output
    'parallel_processing': True,  # Whether to use parallel processing
    'max_workers': 4  # Maximum number of parallel workers
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'era5_processing.log',
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# File naming conventions
FILE_NAMING = {
    'download_pattern': '{variable}_{year}_{month:02d}_{day:02d}_{hour:02d}.{format}',
    'processed_pattern': '{variable}_{year}_{month:02d}_{day:02d}_{hour:02d}_processed.{format}',
    'search_pattern': '{variable}_{bbox}_{start_date}_{end_date}.{format}',
    'metadata_pattern': '{variable}_{year}_{month:02d}_metadata.json'
}

# Grid configuration (from GeoJSON)
GRID_CONFIG = {
    'geojson_path': BASE_DIR / 'ERA5' / 'Grid_50K_MatchedDates.geojson',
    'grid_id_field': 'id',
    'phien_hieu_field': 'PhienHieu',
    'dates_field': 'm_dates',
    'count_field': 'm_count'
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'chunk_size': 1000,  # Number of files to process in each chunk
    'memory_limit': '4GB',  # Memory limit for processing
    'temp_dir': BASE_DIR / 'temp',  # Temporary directory for processing
    'cleanup_temp': True,  # Whether to cleanup temporary files
    'cache_results': True,  # Whether to cache intermediate results
    'cache_dir': BASE_DIR / 'cache'  # Cache directory
}

# Validation configuration
VALIDATION_CONFIG = {
    'check_data_integrity': True,
    'validate_coordinates': True,
    'check_file_sizes': True,
    'verify_downloads': True,
    'log_validation_errors': True
}

# Error handling configuration
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 60,
    'continue_on_error': True,
    'log_errors': True,
    'save_error_logs': True,
    'error_log_dir': BASE_DIR / 'error_logs'
}

# Create additional directories
for dir_path in [
    LOGGING_CONFIG['file'].parent,
    PERFORMANCE_CONFIG['temp_dir'],
    PERFORMANCE_CONFIG['cache_dir'],
    ERROR_CONFIG['error_log_dir']
]:
    dir_path.mkdir(parents=True, exist_ok=True) 