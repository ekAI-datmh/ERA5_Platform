#!/usr/bin/env python3
"""
Test script to check extracted NetCDF file reading
"""

import xarray as xr
import os

def test_extracted_netcdf():
    """Test if the extracted NetCDF file can be read"""
    
    filename = 'data_0.nc'
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return False
    
    print(f"File '{filename}' exists. Size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    try:
        # Try to open the dataset
        print("Attempting to open extracted NetCDF dataset...")
        ds = xr.open_dataset(filename)
        
        print("✅ Successfully opened dataset!")
        print(f"Variables: {list(ds.data_vars.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        
        # Print some basic info about the data
        if 'time' in ds.coords:
            print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
        
        if 'latitude' in ds.coords:
            print(f"Latitude range: {ds.latitude.min().values:.2f} to {ds.latitude.max().values:.2f}")
        
        if 'longitude' in ds.coords:
            print(f"Longitude range: {ds.longitude.min().values:.2f} to {ds.longitude.max().values:.2f}")
        
        # Print variable information
        print("\nVariable details:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"  {var_name}: {var.dims} - {var.attrs.get('long_name', 'No description')}")
        
        # Close the dataset
        ds.close()
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

if __name__ == "__main__":
    print("=== Extracted NetCDF File Reading Test ===\n")
    success = test_extracted_netcdf()
    
    if success:
        print("\n✅ Test passed! Your NetCDF file can be read properly.")
        print("\nYou can now use this code in your notebook:")
        print("```python")
        print("import xarray as xr")
        print("ds = xr.open_dataset('data_0.nc')")
        print("```")
        print("\nOr rename the file:")
        print("```bash")
        print("mv data_0.nc era5_land_hourly_202401_extracted.nc")
        print("```")
    else:
        print("\n❌ Test failed! Please check the error message above.") 