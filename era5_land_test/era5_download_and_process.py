#!/usr/bin/env python3
"""
ERA5-Land Data Download and Processing Script
- Downloads ERA5-Land data via CDS API
- Automatically extracts ZIP files if needed
- Resamples data to 30-meter resolution
- Optimized for Vietnam region
"""

import os
import zipfile
import tempfile
import shutil
from pathlib import Path
import xarray as xr
import numpy as np
from era5_data_download import CDSAPIConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_era5_data(output_filename='era5_land_vietnam_202401.nc'):
    """
    Download ERA5-Land data for Vietnam
    
    Args:
        output_filename (str): Name for the output file
    
    Returns:
        str: Path to the downloaded file (may be ZIP)
    """
    try:
        logger.info("Starting ERA5-Land data download...")
        
        # Initialize CDS API client
        config = CDSAPIConfig(env_file='.env')
        client = config.get_client()
        
        # Define request parameters
        request_params = {
            'format': 'netcdf',
            'product_type': 'reanalysis',
            'variable': ['total_precipitation', '2m_temperature'],
            'year': '2024',
            'month': '01',
            'day': list(range(1, 32)),  # All days in January
            'time': [f'{h:02d}:00' for h in range(0, 24, 3)],  # Every 3 hours
            'area': [23.5, 102.0, 8.0, 110.0],  # Vietnam bounding box [N, W, S, E]
        }
        
        logger.info(f"Request parameters: {request_params}")
        logger.info("This may take several minutes...")
        
        # Download the data
        client.retrieve('reanalysis-era5-land', request_params, output_filename)
        
        logger.info(f"Download completed: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def extract_if_zip(file_path):
    """
    Extract ZIP file if needed and return path to NetCDF file
    
    Args:
        file_path (str): Path to the downloaded file
    
    Returns:
        str: Path to the extracted NetCDF file
    """
    # Check if file is a ZIP archive
    if zipfile.is_zipfile(file_path):
        logger.info(f"File {file_path} is a ZIP archive. Extracting...")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contents: {file_list}")
            
            # Find NetCDF file
            nc_files = [f for f in file_list if f.endswith('.nc')]
            if not nc_files:
                raise ValueError("No NetCDF files found in ZIP archive")
            
            # Extract the first NetCDF file
            nc_file = nc_files[0]
            zip_ref.extract(nc_file)
            
            # Rename to a more descriptive name
            extracted_path = f"{Path(file_path).stem}_extracted.nc"
            shutil.move(nc_file, extracted_path)
            
            logger.info(f"Extracted NetCDF file: {extracted_path}")
            return extracted_path
    else:
        logger.info(f"File {file_path} is already a NetCDF file")
        return file_path

def resample_to_30m(input_file, output_file, target_resolution_m=30):
    """
    Resample ERA5-Land data to 30-meter resolution using bilinear interpolation
    
    Args:
        input_file (str): Path to input NetCDF file
        output_file (str): Path to output resampled file
        target_resolution_m (int): Target resolution in meters
    
    Returns:
        str: Path to the resampled file
    """
    logger.info(f"Resampling {input_file} to {target_resolution_m}m resolution...")
    
    try:
        # Open the dataset
        ds = xr.open_dataset(input_file)
        
        # Get the current bounds
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
        
        logger.info(f"Original bounds: lat [{lat_min:.3f}, {lat_max:.3f}], lon [{lon_min:.3f}, {lon_max:.3f}]")
        logger.info(f"Original shape: {ds.latitude.size} x {ds.longitude.size}")
        
        # Calculate target grid size based on 30m resolution
        # 1 degree ≈ 111,319 meters at equator
        meters_per_degree = 111319
        degree_resolution = target_resolution_m / meters_per_degree
        
        # Create new coordinate arrays
        new_lats = np.arange(lat_min, lat_max + degree_resolution, degree_resolution)
        new_lons = np.arange(lon_min, lon_max + degree_resolution, degree_resolution)
        
        # Ensure we don't exceed the original bounds
        new_lats = new_lats[new_lats <= lat_max]
        new_lons = new_lons[new_lons <= lon_max]
        
        logger.info(f"New grid size: {len(new_lats)} x {len(new_lons)} points")
        logger.info(f"Approximate resolution: {degree_resolution:.6f} degrees ({target_resolution_m}m)")
        
        # Interpolate to new grid
        logger.info("Performing bilinear interpolation...")
        
        # Use xarray's interpolation method
        ds_resampled = ds.interp(
            latitude=new_lats,
            longitude=new_lons,
            method='linear'
        )
        
        # Add metadata about the resampling
        ds_resampled.attrs['resampling_info'] = f'Resampled to {target_resolution_m}m resolution using bilinear interpolation'
        ds_resampled.attrs['original_resolution'] = '~11km'
        ds_resampled.attrs['target_resolution'] = f'{target_resolution_m}m'
        ds_resampled.attrs['resampling_method'] = 'bilinear'
        
        # Update coordinate attributes
        ds_resampled.latitude.attrs['resolution_m'] = target_resolution_m
        ds_resampled.longitude.attrs['resolution_m'] = target_resolution_m
        
        # Save the resampled dataset
        logger.info(f"Saving resampled data to {output_file}...")
        
        # Use compression to reduce file size
        encoding = {}
        for var in ds_resampled.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 4}
        
        ds_resampled.to_netcdf(output_file, encoding=encoding)
        
        # Close datasets
        ds.close()
        ds_resampled.close()
        
        logger.info(f"Resampling completed: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        raise

def download_and_process_era5(
    year='2024',
    month='01',
    days=None,
    variables=['total_precipitation', '2m_temperature'],
    target_resolution_m=30,
    output_prefix='era5_vietnam'
):
    """
    Complete workflow: download, extract, and resample ERA5-Land data
    
    Args:
        year (str): Year to download
        month (str): Month to download (format: '01', '02', etc.)
        days (list): List of days to download (default: all days in month)
        variables (list): Variables to download
        target_resolution_m (int): Target resolution in meters
        output_prefix (str): Prefix for output files
    
    Returns:
        dict: Paths to generated files
    """
    if days is None:
        if month in ['01', '03', '05', '07', '08', '10', '12']:
            days = list(range(1, 32))
        elif month in ['04', '06', '09', '11']:
            days = list(range(1, 31))
        else:  # February
            days = list(range(1, 29))  # Assuming non-leap year
    
    # File names
    raw_file = f"{output_prefix}_{year}{month}_raw.nc"
    extracted_file = f"{output_prefix}_{year}{month}_extracted.nc"
    resampled_file = f"{output_prefix}_{year}{month}_{target_resolution_m}m.nc"
    
    try:
        # Step 1: Download
        logger.info("=" * 50)
        logger.info("STEP 1: Downloading ERA5-Land data")
        logger.info("=" * 50)
        downloaded_file = download_era5_data(raw_file)
        
        # Step 2: Extract if ZIP
        logger.info("=" * 50)
        logger.info("STEP 2: Extracting data if needed")
        logger.info("=" * 50)
        netcdf_file = extract_if_zip(downloaded_file)
        
        # Rename extracted file for clarity
        if netcdf_file != downloaded_file:
            shutil.move(netcdf_file, extracted_file)
            netcdf_file = extracted_file
        
        # Step 3: Resample to 30m
        logger.info("=" * 50)
        logger.info(f"STEP 3: Resampling to {target_resolution_m}m resolution")
        logger.info("=" * 50)
        final_file = resample_to_30m(netcdf_file, resampled_file, target_resolution_m)
        
        # Results
        results = {
            'raw_download': downloaded_file,
            'extracted_netcdf': netcdf_file,
            'resampled_30m': final_file
        }
        
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 50)
        for key, path in results.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"{key}: {path} ({size_mb:.2f} MB)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in processing workflow: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    results = download_and_process_era5(
        year='2024',
        month='01',
        target_resolution_m=30,
        output_prefix='vietnam_era5'
    ) 