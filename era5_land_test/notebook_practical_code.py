"""
PRACTICAL ERA5 RESAMPLING FOR NOTEBOOK
Choose a reasonable resolution that won't crash your system!
"""

import zipfile
import xarray as xr
import numpy as np
from era5_data_download import CDSAPIConfig
import shutil
import os

# CHOOSE YOUR RESOLUTION HERE:
# Options: "5km", "2km", "1km", "500m", "100m"
TARGET_RESOLUTION = "1km"  # 👈 CHANGE THIS

def get_resolution_info():
    """Show what each resolution means"""
    resolutions = {
        "5km": {"meters": 5000, "grid_approx": "3 × 2", "memory": "~0.01 GB", "desc": "Fast, small files"},
        "2km": {"meters": 2000, "grid_approx": "8 × 4", "memory": "~0.06 GB", "desc": "Good balance"},
        "1km": {"meters": 1000, "grid_approx": "16 × 8", "memory": "~0.24 GB", "desc": "High detail, manageable"},
        "500m": {"meters": 500, "grid_approx": "31 × 16", "memory": "~0.95 GB", "desc": "Very high detail"},
        "100m": {"meters": 100, "grid_approx": "155 × 80", "memory": "~24 GB", "desc": "Extreme detail, large files"}
    }
    
    print("🎯 RESOLUTION OPTIONS:")
    print("=" * 50)
    for res, info in resolutions.items():
        print(f"{res:>5}: {info['desc']}")
        print(f"       ~{info['grid_approx']}k points, {info['memory']} memory")
    print("=" * 50)
    
    return resolutions

# Show options
resolution_options = get_resolution_info()
print(f"✅ Selected: {TARGET_RESOLUTION} ({resolution_options[TARGET_RESOLUTION]['desc']})")

# Step 1: Download data (use existing if available)
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
    print("📁 Using existing file...")

# Step 2: Extract ZIP if needed
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
print(f"🔍 Resampling to {TARGET_RESOLUTION} resolution...")

# Get target resolution in meters
target_meters = resolution_options[TARGET_RESOLUTION]["meters"]

# Open dataset
ds = xr.open_dataset(netcdf_file)

# Calculate grid
meters_per_degree = 111319
degree_resolution = target_meters / meters_per_degree

lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())

new_lats = np.arange(lat_min, lat_max, degree_resolution)
new_lons = np.arange(lon_min, lon_max, degree_resolution)

print(f"Original: {ds.latitude.size} × {ds.longitude.size}")
print(f"New grid: {len(new_lats):,} × {len(new_lons):,}")

# Memory estimate
total_points = len(new_lats) * len(new_lons) * len(ds.valid_time) * 2
memory_gb = (total_points * 8) / (1024**3)
print(f"Estimated memory: {memory_gb:.2f} GB")

# Use chunking for large grids
if len(new_lats) * len(new_lons) > 1_000_000:
    print("🔄 Using memory-efficient chunking...")
    ds = ds.chunk({'latitude': 50, 'longitude': 50})

# Interpolate
print("🔄 Interpolating...")
ds_resampled = ds.interp(latitude=new_lats, longitude=new_lons, method='linear')

# Add metadata
ds_resampled.attrs['resolution'] = f'{target_meters}m'
ds_resampled.attrs['resampling_method'] = 'bilinear'

# Save
output_file = f'era5_vietnam_{TARGET_RESOLUTION}.nc'
print(f"💾 Saving to {output_file}...")

# Compression
encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_resampled.data_vars}
ds_resampled.to_netcdf(output_file, encoding=encoding)

# Results
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"✅ COMPLETE!")
print(f"📁 File: {output_file}")
print(f"📊 Size: {file_size_mb:.1f} MB")
print(f"🎯 Resolution: {TARGET_RESOLUTION} (~{target_meters}m)")

ds.close()
ds_resampled.close()

# Quick verification
print("\n🔍 Quick check:")
ds_check = xr.open_dataset(output_file)
print(f"Variables: {list(ds_check.data_vars.keys())}")
print(f"Grid: {ds_check.latitude.size} × {ds_check.longitude.size}")
print(f"Time steps: {ds_check.valid_time.size}")
ds_check.close() 