#!/usr/bin/env python3
"""
Practical ERA5 resampling with memory-efficient options
"""

import zipfile
import xarray as xr
import numpy as np
from era5_data_download import CDSAPIConfig
import shutil

def estimate_memory_usage(lat_size, lon_size, time_size, num_variables=2):
    """Estimate memory usage for a given grid size"""
    total_points = lat_size * lon_size * time_size * num_variables
    # Each float64 point = 8 bytes
    memory_gb = (total_points * 8) / (1024**3)
    return memory_gb

def get_practical_resolution_options():
    """Get practical resolution options for Vietnam"""
    # Vietnam bounds
    lat_range = 23.5 - 8.0  # ~15.5 degrees
    lon_range = 110.0 - 102.0  # ~8 degrees
    
    options = {
        "1km": {"meters": 1000, "description": "1 kilometer - High detail, manageable size"},
        "500m": {"meters": 500, "description": "500 meters - Very high detail, large files"},
        "100m": {"meters": 100, "description": "100 meters - Extremely high detail, very large"},
        "2km": {"meters": 2000, "description": "2 kilometers - Good balance"},
        "5km": {"meters": 5000, "description": "5 kilometers - Fast processing, smaller files"}
    }
    
    print("Available resolution options for Vietnam:")
    print("=" * 60)
    
    for name, info in options.items():
        meters_per_degree = 111319
        degree_res = info["meters"] / meters_per_degree
        
        lat_points = int(lat_range / degree_res)
        lon_points = int(lon_range / degree_res)
        
        # Estimate memory (assuming 248 time steps, 2 variables)
        memory_gb = estimate_memory_usage(lat_points, lon_points, 248, 2)
        
        print(f"{name:>6}: {info['description']}")
        print(f"        Grid: {lat_points:,} × {lon_points:,} = {lat_points * lon_points:,} points")
        print(f"        Memory: ~{memory_gb:.2f} GB")
        print()
    
    return options

def download_and_resample_practical(target_resolution="1km"):
    """
    Download and resample to practical resolution
    """
    
    # Show available options
    options = get_practical_resolution_options()
    
    if target_resolution not in options:
        print(f"Error: '{target_resolution}' not available. Choose from: {list(options.keys())}")
        return None
    
    resolution_info = options[target_resolution]
    target_meters = resolution_info["meters"]
    
    print(f"Selected resolution: {target_resolution} ({target_meters}m)")
    print("=" * 50)
    
    # Step 1: Download (if not already downloaded)
    downloaded_file = 'era5_vietnam_raw.nc'
    if not os.path.exists(downloaded_file):
        print("📥 Downloading ERA5-Land data...")
        c = CDSAPIConfig(env_file='.env').get_client()
        c.retrieve(
            'reanalysis-era5-land',
            {
                'format': 'netcdf',
                'product_type': 'reanalysis',
                'variable': ['total_precipitation', '2m_temperature'],
                'year': '2024',
                'month': '01',
                'day': list(range(1, 32)),
                'time': [f'{h:02d}:00' for h in range(0, 24, 3)],
                'area': [23.5, 102.0, 8.0, 110.0],  # Vietnam
            },
            downloaded_file
        )
    else:
        print("📁 Using existing downloaded file...")
    
    # Step 2: Extract if ZIP
    if zipfile.is_zipfile(downloaded_file):
        print("🔓 Extracting ZIP file...")
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
            zip_ref.extract(nc_files[0])
            shutil.move(nc_files[0], 'era5_vietnam_extracted.nc')
            netcdf_file = 'era5_vietnam_extracted.nc'
    else:
        netcdf_file = downloaded_file
    
    # Step 3: Smart resampling
    print(f"🔍 Resampling to {target_resolution} resolution...")
    
    ds = xr.open_dataset(netcdf_file)
    
    # Calculate new grid
    meters_per_degree = 111319
    degree_resolution = target_meters / meters_per_degree
    
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    
    # Create new coordinates with reasonable bounds
    new_lats = np.arange(lat_min, lat_max, degree_resolution)
    new_lons = np.arange(lon_min, lon_max, degree_resolution)
    
    print(f"Original grid: {ds.latitude.size} × {ds.longitude.size}")
    print(f"New grid: {len(new_lats):,} × {len(new_lons):,}")
    
    # Memory check
    memory_gb = estimate_memory_usage(len(new_lats), len(new_lons), len(ds.valid_time), 2)
    print(f"Estimated memory: {memory_gb:.2f} GB")
    
    if memory_gb > 32:  # Warning for >32GB
        print("⚠️  WARNING: This will require significant memory!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            ds.close()
            return None
    
    # Process in chunks for large grids
    if len(new_lats) * len(new_lons) > 10_000_000:  # >10M points
        print("🔄 Using chunked processing for large grid...")
        # Use dask chunks for memory efficiency
        ds = ds.chunk({'latitude': 50, 'longitude': 50, 'valid_time': -1})
    
    # Interpolate
    print("🔄 Interpolating...")
    ds_resampled = ds.interp(
        latitude=new_lats, 
        longitude=new_lons, 
        method='linear'
    )
    
    # Add metadata
    ds_resampled.attrs['resolution'] = f'{target_meters}m'
    ds_resampled.attrs['resampling_method'] = 'bilinear_interpolation'
    ds_resampled.attrs['original_resolution'] = '~11km (ERA5-Land)'
    
    # Save with optimized settings
    output_file = f'era5_vietnam_{target_resolution}.nc'
    print(f"💾 Saving to {output_file}...")
    
    # Compression settings based on grid size
    if len(new_lats) * len(new_lons) > 1_000_000:
        # High compression for large files
        encoding = {var: {'zlib': True, 'complevel': 9, 'shuffle': True} 
                   for var in ds_resampled.data_vars}
    else:
        # Standard compression
        encoding = {var: {'zlib': True, 'complevel': 4} 
                   for var in ds_resampled.data_vars}
    
    ds_resampled.to_netcdf(output_file, encoding=encoding)
    
    # File size info
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Complete! File: {output_file} ({file_size_mb:.1f} MB)")
    
    ds.close()
    ds_resampled.close()
    
    return output_file

# Import os for file operations
import os

if __name__ == "__main__":
    # Show options first
    get_practical_resolution_options()
    
    # Example usage - start with 1km resolution
    print("Starting with 1km resolution (recommended)...")
    result = download_and_resample_practical("1km") 