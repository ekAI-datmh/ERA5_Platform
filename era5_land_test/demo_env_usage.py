
import os
import cdsapi
from typing import Optional, Dict, Any
from dotenv import load_dotenv

class CDSAPIConfig:
    """Configuration class for CDS API with multiple credential sources"""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize CDS API configuration
        
        Args:
            url: CDS API URL (optional, will use environment variable or default)
            key: CDS API key (optional, will use environment variable)
            env_file: Path to .env file to load (optional)
        """
        # Load .env file if specified
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load .env file from current directory
            load_dotenv()
        
        self.url = url or os.getenv('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api')
        self.key = key or os.getenv('CDSAPI_KEY')
        
        if not self.key:
            raise ValueError("CDS API key is required. Set CDSAPI_KEY environment variable or pass key parameter.")
    
    def get_client(self) -> cdsapi.Client:
        """Create and return a configured CDS API client"""
        return cdsapi.Client(url=self.url, key=self.key)


import zipfile
import xarray as xr
import numpy as np
from era5_data_download import CDSAPIConfig
import shutil
import os
import calendar # To get the number of days in each month
from pathlib import Path # Import Path for easier directory management

# Initialize CDS API client
c = CDSAPIConfig(env_file='.env').get_client()

year = 2024 # You can change this to any year you need

# Define the base output directory
base_output_dir = Path('data')

print(f"Starting ERA5-Land data download for year {year} (daily files)...")
print("-" * 60)

for month in range(1, 13): # Loop through all 12 months
    # Determine the number of days in the current month
    _, num_days = calendar.monthrange(year, month)

    # Define the output directory for the current month and create it if it doesn't exist
    month_output_dir = base_output_dir / str(year) / f"{month:02d}"
    month_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory: {month_output_dir}")

    for day in range(1, num_days + 1): # Loop through all days in the month
        # Define output filenames with the new directory structure
        output_filename_raw = month_output_dir / f'era5_vietnam_raw_{year}_{month:02d}_{day:02d}.nc'
        output_filename_extracted = month_output_dir / f'era5_vietnam_{year}_{month:02d}_{day:02d}.nc'

        print(f"Downloading data for {year}-{month:02d}-{day:02d} to {month_output_dir}...")

        try:
            c.retrieve(
                'reanalysis-era5-land',
                #########################################################
                # Sai ten bands
                {
                    'format': 'netcdf',
                    'product_type': 'reanalysis',
                    'variable': [
                        "2m_dewpoint_temperature",
                        "2m_temperature",
                        "skin_temperature",
                        "surface_latent_heat_flux",
                        "surface_net_solar_radiation",
                        "surface_sensible_heat_flux",
                        "surface_solar_radiation_downwards",
                        "potential_evaporation",
                        "total_evaporation",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "surface_pressure",
                        "total_precipitation"
                        ],
                    'year': str(year),
                    'month': f'{month:02d}',
                    'day': [f'{day:02d}'], # Request single day
                    'time': [f'{h:02d}:00' for h in range(0, 24)],
                    'area': [23.5, 102.0, 8.0, 110.0],  # Vietnam bounding box
                    # 'grid': [0.1, 0.1] # 0.1 degree resolution (~11km) - uncomment if needed
                },
                str(output_filename_raw) # Convert Path object to string for retrieve
            )
            print(f"Download complete: {output_filename_raw}")

            # Auto-extract ZIP if needed
            if zipfile.is_zipfile(output_filename_raw):
                print(f"Extracting ZIP file: {output_filename_raw}...")
                with zipfile.ZipFile(output_filename_raw, 'r') as zip_ref:
                    nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                    if nc_files:
                        # Extract the first .nc file found in the zip to the current directory
                        zip_ref.extract(nc_files[0], path=month_output_dir) # Extract to month_output_dir
                        
                        # Move the extracted file to its final unique filename within the month directory
                        extracted_temp_path = month_output_dir / nc_files[0]
                        shutil.move(extracted_temp_path, output_filename_extracted)
                        print(f"Extracted to {output_filename_extracted}")
                    else:
                        print(f"No NetCDF file found inside {output_filename_raw}")
                # Remove the raw zip file after extraction
                os.remove(output_filename_raw)
            else:
                # If it's already a .nc file, and not a zip, just rename it
                shutil.move(output_filename_raw, output_filename_extracted)
                print(f"Downloaded directly as NetCDF: {output_filename_extracted}")

        except Exception as e:
            print(f"Error processing {year}-{month:02d}-{day:02d}: {e}")
            continue # Continue to the next day even if one fails

print("-" * 60)
print(f"Finished processing all data for year {year}.")


if __name__ == "__main__":
    demo_env_usage() 