#!/usr/bin/env python3
"""
Test script to check NetCDF file reading with explicit engine
"""

import xarray as xr
import os

def test_netcdf_reading():
    """Test if the NetCDF file can be read with explicit engine"""
    
    filename = 'era5_land_hourly_202401.nc'
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return False
    
    print(f"File '{filename}' exists. Size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    try:
        # Try to open the dataset with explicit engine
        print("Attempting to open dataset with explicit netcdf4 engine...")
        ds = xr.open_dataset(filename, engine='netcdf4')
        
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
        
        # Close the dataset
        ds.close()
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

if __name__ == "__main__":
    print("=== NetCDF File Reading Test (Explicit Engine) ===\n")
    success = test_netcdf_reading()
    
    if success:
        print("\n✅ Test passed! Your NetCDF file can be read properly.")
        print("\nYou can now use this code in your notebook:")
        print("```python")
        print("import xarray as xr")
        print("ds = xr.open_dataset('era5_land_hourly_202401.nc', engine='netcdf4')")
        print("```")
    else:
        print("\n❌ Test failed! Please check the error message above.") 