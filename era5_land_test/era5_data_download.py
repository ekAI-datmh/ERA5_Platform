#!/usr/bin/env python3
"""
ERA5-Land data download script using CDS API
Demonstrates multiple ways to configure CDS API credentials outside of .cdsapirc
"""

import os
import cdsapi
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def download_era5_land_data(
    variable: str,
    year: int,
    month: int,
    day: int,
    time: str,
    area: list,
    output_file: str,
    config: Optional[CDSAPIConfig] = None
) -> bool:
    """
    Download ERA5-Land data using CDS API
    
    Args:
        variable: Variable to download (e.g., '2m_temperature')
        year: Year for data
        month: Month for data
        day: Day for data
        time: Time in format 'HH:MM' or list of times
        area: Geographic area [north, west, south, east]
        output_file: Output file path
        config: CDS API configuration (optional)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Use provided config or create from environment variables
        if config is None:
            config = CDSAPIConfig()
        
        client = config.get_client()
        
        # Prepare request parameters
        request_params = {
            'product_type': 'reanalysis',
            'variable': variable,
            'year': str(year),
            'month': f"{month:02d}",
            'day': f"{day:02d}",
            'time': time if isinstance(time, list) else [time],
            'area': area,
            'format': 'netcdf',
        }
        
        logger.info(f"Downloading {variable} for {year}-{month:02d}-{day:02d}")
        logger.info(f"Request parameters: {request_params}")
        
        # Make the request
        client.retrieve('reanalysis-era5-land', request_params, output_file)
        
        logger.info(f"Successfully downloaded data to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False


def example_usage():
    """Example usage showing different configuration methods"""
    
    # Method 1: Using environment variables
    print("Method 1: Using environment variables")
    print("Set these environment variables:")
    print("export CDSAPI_URL='https://cds.climate.copernicus.eu/api'")
    print("export CDSAPI_KEY='your-api-key-here'")
    
    # Method 2: Using .env file
    print("\nMethod 2: Using .env file")
    print("Create a .env file with:")
    print("CDSAPI_URL=https://cds.climate.copernicus.eu/api")
    print("CDSAPI_KEY=your-api-key-here")
    
    # Method 3: Direct configuration
    print("\nMethod 3: Direct configuration")
    config = CDSAPIConfig(
        url='https://cds.climate.copernicus.eu/api',
        key='your-api-key-here'
    )
    
    # Method 4: Reading from custom config file
    print("\nMethod 4: Reading from custom config file")
    
    def load_config_from_file(config_file: str) -> CDSAPIConfig:
        """Load CDS API configuration from a custom file"""
        config_data = {}
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    config_data[key.strip()] = value.strip()
        
        return CDSAPIConfig(
            url=config_data.get('url'),
            key=config_data.get('key')
        )
    
    # Example downloads
    print("\nExample downloads:")
    
    # Example 1: Using .env file (automatically loaded)
    print("\nExample 1: Using .env file")
    try:
        success = download_era5_land_data(
            variable='2m_temperature',
            year=2023,
            month=1,
            day=1,
            time=['00:00', '06:00', '12:00', '18:00'],
            area=[60, -10, 50, 2],  # Europe
            output_file='era5_land_temp_20230101.nc'
        )
        
        if success:
            print("Download completed successfully!")
        else:
            print("Download failed!")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using specific .env file
    print("\nExample 2: Using specific .env file")
    try:
        config = CDSAPIConfig(env_file='./custom.env')
        success = download_era5_land_data(
            variable='2m_temperature',
            year=2023,
            month=1,
            day=1,
            time=['00:00', '06:00', '12:00', '18:00'],
            area=[60, -10, 50, 2],  # Europe
            output_file='era5_land_temp_20230101_v2.nc',
            config=config
        )
        
        if success:
            print("Download completed successfully!")
        else:
            print("Download failed!")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
