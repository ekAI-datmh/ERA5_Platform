#!/usr/bin/env python3
"""
Simplified ERA5 download and processing for notebook use
"""

import zipfile
import xarray as xr
import numpy as np
from era5_data_download import CDSAPIConfig

def download_and_process_era5_simple():
    """
    Download ERA5 data, extract if ZIP, and resample to 30m
    """
    
    # Step 1: Download data
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
        'era5_land_vietnam_raw.nc'
    )
    
    # Step 2: Extract if ZIP
    print("📂 Checking if file needs extraction...")
    downloaded_file = 'era5_land_vietnam_raw.nc'
    
    if zipfile.is_zipfile(downloaded_file):
        print("🔓 Extracting ZIP file...")
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
            if nc_files:
                zip_ref.extract(nc_files[0])
                # Rename extracted file
                import shutil
                shutil.move(nc_files[0], 'era5_land_vietnam_extracted.nc')
                netcdf_file = 'era5_land_vietnam_extracted.nc'
            else:
                raise ValueError("No NetCDF files in ZIP")
    else:
        netcdf_file = downloaded_file
    
    # Step 3: Resample to 30m
    print("🔍 Resampling to 30-meter resolution...")
    
    # Open dataset
    ds = xr.open_dataset(netcdf_file)
    
    # Calculate new grid (30m resolution)
    meters_per_degree = 111319  # Approximate at equator
    degree_resolution = 30 / meters_per_degree  # 30m in degrees
    
    # Get bounds
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    
    # Create new coordinates
    new_lats = np.arange(lat_min, lat_max, degree_resolution)
    new_lons = np.arange(lon_min, lon_max, degree_resolution)
    
    print(f"Original grid: {ds.latitude.size} x {ds.longitude.size}")
    print(f"New grid: {len(new_lats)} x {len(new_lons)}")
    print(f"Resolution: ~{degree_resolution:.6f} degrees (~30m)")
    
    # Interpolate
    ds_30m = ds.interp(
        latitude=new_lats,
        longitude=new_lons,
        method='linear'
    )
    
    # Add metadata
    ds_30m.attrs['resolution'] = '30m'
    ds_30m.attrs['resampling_method'] = 'bilinear_interpolation'
    
    # Save
    output_file = 'era5_vietnam_30m.nc'
    print(f"💾 Saving to {output_file}...")
    
    # Compress to save space
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_30m.data_vars}
    ds_30m.to_netcdf(output_file, encoding=encoding)
    
    ds.close()
    ds_30m.close()
    
    print("✅ Complete! Files created:")
    print(f"  - Raw download: {downloaded_file}")
    print(f"  - Extracted: {netcdf_file}")
    print(f"  - 30m resampled: {output_file}")
    
    return output_file

# For notebook use:
if __name__ == "__main__":
    final_file = download_and_process_era5_simple()
    
    # Quick verification
    print("\n🔍 Quick verification:")
    ds = xr.open_dataset(final_file)
    print(f"Variables: {list(ds.data_vars.keys())}")
    print(f"Shape: {ds.latitude.size} x {ds.longitude.size}")
    print(f"Coverage: {ds.latitude.min().values:.3f}°N to {ds.latitude.max().values:.3f}°N")
    print(f"         {ds.longitude.min().values:.3f}°E to {ds.longitude.max().values:.3f}°E")
    ds.close() 