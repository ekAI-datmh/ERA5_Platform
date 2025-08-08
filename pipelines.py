import cdsapi
import os
import json
import random
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(filename='era5_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def read_geojson_grids(geojson_path, sample_size=10):
    """
    Read GeoJSON file and extract sample grids.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
        sample_size (int): Number of sample grids to extract.
    
    Returns:
        list: List of grid dictionaries with coordinates and properties.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    sample_grids = random.sample(features, min(sample_size, len(features)))
    
    grids = []
    for feature in sample_grids:
        coords = feature['geometry']['coordinates'][0]
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        
        grid_info = {
            'id': feature['id'],
            'phien_hieu': feature['properties']['PhienHieu'],
            'min_lon': min(lons),
            'max_lon': max(lons),
            'min_lat': min(lats),
            'max_lat': max(lats),
            'center_lon': sum(lons) / len(lons),
            'center_lat': sum(lats) / len(lats),
            'm_count': feature['properties']['m_count'],
            'm_dates': feature['properties']['m_dates']
        }
        grids.append(grid_info)
    
    return grids

def generate_date_ranges(start_date, end_date):
    """
    Generate a list of year-month tuples for the given date range.
    
    Args:
        start_date (datetime): Starting date.
        end_date (datetime): Ending date.
    
    Returns:
        list: List of (year, month) tuples.
    """
    date_ranges = []
    current_date = start_date
    while current_date <= end_date:
        date_ranges.append((current_date.year, f"{current_date.month:02d}"))
        next_month = current_date.month % 12 + 1
        next_year = current_date.year + (current_date.month // 12)
        current_date = current_date.replace(year=next_year, month=next_month, day=1)
    return date_ranges

def download_era5_data(dataset, request, target, retries=2):
    """
    Download ERA5 data from CDS with retries.
    
    Args:
        dataset (str): CDS dataset name.
        request (dict): Request parameters.
        target (str): Output file path.
        retries (int): Number of retry attempts.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        client = cdsapi.Client()
        for attempt in range(retries):
            try:
                logging.info(f"Attempt {attempt + 1}: Downloading {target}...")
                start_time = time.time()
                client.retrieve(dataset, request, target)
                end_time = time.time()
                
                if os.path.exists(target):
                    file_size = os.path.getsize(target) / (1024 * 1024)  # MB
                    download_time = end_time - start_time
                    logging.info(f"Success: {target} ({file_size:.2f} MB, {download_time:.2f}s)")
                    return True
                else:
                    logging.warning(f"Failed to download {target}")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Error downloading {target}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error initializing client for {target}: {str(e)}")
        return False

def main():
    # Configuration
    geojson_path = 'ERA5/Grid_50K_MatchedDates.geojson'
    sample_size = 3  # Very small sample for testing
    start_date = datetime(2024, 1, 1)  # Recent date range
    end_date = datetime(2024, 1, 1)    # Just one month for testing
    data_dir = 'data'
    
    # Read sample grids from GeoJSON
    logging.info(f"Reading {sample_size} sample grids from {geojson_path}...")
    grids = read_geojson_grids(geojson_path, sample_size)
    
    # Generate date ranges
    date_ranges = generate_date_ranges(start_date, end_date)
    
    # Use only variables that we know work from the test script
    working_variables = ['2m_temperature', 'surface_pressure', 'total_precipitation']
    
    # Create data directory structure
    os.makedirs(os.path.join(data_dir, 'era5_test', 'grib'), exist_ok=True)
    
    # Download data for each grid
    for grid in grids:
        area = [grid['max_lat'], grid['min_lon'], grid['min_lat'], grid['max_lon']]
        grid_id = grid['id']
        
        for year, month in date_ranges:
            # Define request with only working variables
            request = {
                'product_type': 'reanalysis',
                'variable': working_variables,
                'year': [str(year)],
                'month': [month],
                'day': ['01', '02', '03'],  # Just a few days for testing
                'time': ['00:00', '06:00', '12:00', '18:00'],  # 4 times per day
                'area': area,
                'format': 'grib'
            }
            
            # Target file path
            target_dir = os.path.join(data_dir, 'era5_test', 'grib', str(year), month)
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, f"{grid_id}_{year}_{month}.grib")
            
            # Skip if file already exists
            if os.path.exists(target_file):
                logging.info(f"Skipping existing file: {target_file}")
                continue
            
            # Download data
            success = download_era5_data('reanalysis-era5-single-levels', request, target_file)
            if success:
                logging.info(f"Downloaded {target_file}")
            else:
                logging.error(f"Failed to download {target_file}")
                
            # Add delay between requests to avoid overwhelming the API
            time.sleep(3)

if __name__ == "__main__":
    main()