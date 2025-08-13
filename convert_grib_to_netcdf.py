#!/usr/bin/env python3
"""
Convert GRIB files to NetCDF format for CDO compatibility
"""

import os
import sys
from pathlib import Path
import pygrib
import xarray as xr
import numpy as np


def convert_grib_to_netcdf(grib_file, output_file=None):
    """
    Convert a GRIB file to NetCDF format.
    
    Args:
        grib_file: Path to input GRIB file
        output_file: Path to output NetCDF file (optional)
    
    Returns:
        Path to output NetCDF file
    """
    grib_file = Path(grib_file)
    
    if output_file is None:
        output_file = grib_file.with_suffix('.nc')
    
    print(f"Converting {grib_file} to {output_file}")
    
    try:
        # Open GRIB file
        grbs = pygrib.open(str(grib_file))
        
        # Get the first message
        grb = grbs[1]
        
        # Extract data and coordinates
        data = grb.values
        lats, lons = grb.latlons()
        
        # Get metadata
        variable_name = grb.name.replace(' ', '_').lower()
        units = grb.units
        level = grb.level
        time = grb.validDate
        
        print(f"Variable: {grb.name}")
        print(f"Units: {units}")
        print(f"Level: {level}")
        print(f"Time: {time}")
        print(f"Data shape: {data.shape}")
        print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}")
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                variable_name: xr.DataArray(
                    data=data,
                    dims=['latitude', 'longitude'],
                    attrs={
                        'units': units,
                        'long_name': grb.name,
                        'standard_name': variable_name
                    }
                )
            },
            coords={
                'latitude': xr.DataArray(
                    lats[:, 0],  # Take first column
                    dims=['latitude'],
                    attrs={'units': 'degrees_north', 'standard_name': 'latitude'}
                ),
                'longitude': xr.DataArray(
                    lons[0, :],  # Take first row
                    dims=['longitude'],
                    attrs={'units': 'degrees_east', 'standard_name': 'longitude'}
                ),
                'time': xr.DataArray(
                    [time],
                    dims=['time'],
                    attrs={'standard_name': 'time'}
                )
            },
            attrs={
                'title': f'ERA5_Land {grb.name}',
                'source': 'ERA5_Land reanalysis',
                'history': f'Converted from GRIB using pygrib and xarray'
            }
        )
        
        # Save to NetCDF
        ds.to_netcdf(output_file, format='NETCDF4')
        
        print(f"Successfully converted to {output_file}")
        grbs.close()
        
        return output_file
        
    except Exception as e:
        print(f"Error converting {grib_file}: {e}")
        return None


def main():
    """Convert all GRIB files in the data directory."""
    grib_dir = Path("data/era5_vietnam/grib/2024/01")
    output_dir = Path("data/era5_vietnam/netcdf/2024/01")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find our downloaded GRIB files
    grib_files = list(grib_dir.glob("2m_temperature_*.grib")) + list(grib_dir.glob("total_precipitation_*.grib"))
    
    print(f"Found {len(grib_files)} GRIB files to convert")
    
    for grib_file in grib_files:
        output_file = output_dir / f"{grib_file.stem}.nc"
        result = convert_grib_to_netcdf(grib_file, output_file)
        
        if result:
            print(f"✅ Converted: {grib_file.name} -> {result.name}")
        else:
            print(f"❌ Failed: {grib_file.name}")


if __name__ == "__main__":
    main() 