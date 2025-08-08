import cdsapi
import os
import json
import random
from datetime import datetime
import time

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
    
    # Extract all features
    features = data['features']
    
    # Randomly sample grids
    sample_grids = random.sample(features, min(sample_size, len(features)))
    
    grids = []
    for feature in sample_grids:
        # Extract coordinates from the polygon
        coords = feature['geometry']['coordinates'][0]  # First ring of polygon
        
        # Calculate bounding box
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
                print(f"Attempt {attempt + 1}: Downloading {target}...")
                start_time = time.time()
                client.retrieve(dataset, request, target)
                end_time = time.time()
                
                if os.path.exists(target):
                    file_size = os.path.getsize(target) / (1024 * 1024)  # MB
                    download_time = end_time - start_time
                    print(f"Success: {target} ({file_size:.2f} MB, {download_time:.2f}s)")
                    return True, file_size, download_time
                else:
                    print(f"Failed to download {target}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error downloading {target}: {str(e)}")
        return False, 0, 0
    except Exception as e:
        print(f"Error initializing client for {target}: {str(e)}")
        return False, 0, 0

def main():
    # Configuration
    geojson_path = 'ERA5/Grid_50K_MatchedDates.geojson'
    sample_size = 10
    
    # Read sample grids from GeoJSON
    print(f"Reading {sample_size} sample grids from {geojson_path}...")
    grids = read_geojson_grids(geojson_path, sample_size)
    
    print(f"Selected {len(grids)} grids for testing:")
    for i, grid in enumerate(grids):
        print(f"  {i+1}. {grid['phien_hieu']} (ID: {grid['id']}) - "
              f"Area: [{grid['min_lat']:.3f}, {grid['max_lat']:.3f}] x "
              f"[{grid['min_lon']:.3f}, {grid['max_lon']:.3f}]")
    
    # Test configuration
    test_config = {
        'year': '2024',
        'month': '01',
        'day': '15',
        'time': '12:00',
        'variables': [
            '2m_temperature', 'surface_pressure', 'total_precipitation'
        ],
        'datasets': {
            'grib': 'reanalysis-era5-single-levels',
            'netcdf': 'reanalysis-era5-single-levels'
        }
    }
    
    # Results storage
    results = {
        'grib': {'success': 0, 'total_size': 0, 'total_time': 0, 'files': []},
        'netcdf': {'success': 0, 'total_size': 0, 'total_time': 0, 'files': []}
    }
    
    # Download data for each grid in both formats
    for i, grid in enumerate(grids):
        print(f"\n--- Processing Grid {i+1}/{len(grids)}: {grid['phien_hieu']} ---")
        
        # Define area for this grid
        area = [grid['max_lat'], grid['min_lon'], grid['min_lat'], grid['max_lon']]
        
        # Test GRIB format
        grib_request = {
            'product_type': 'reanalysis',
            'variable': test_config['variables'],
            'year': [test_config['year']],
            'month': [test_config['month']],
            'day': [test_config['day']],
            'time': [test_config['time']],
            'area': area,
            'format': 'grib'
        }
        
        grib_target = f'era5_test_grib_grid_{grid["id"]}.grib'
        success, size, download_time = download_era5_data(
            test_config['datasets']['grib'], grib_request, grib_target
        )
        
        if success:
            results['grib']['success'] += 1
            results['grib']['total_size'] += size
            results['grib']['total_time'] += download_time
            results['grib']['files'].append({
                'grid_id': grid['id'],
                'file': grib_target,
                'size_mb': size,
                'time_s': download_time
            })
        
        # Test NetCDF format
        netcdf_request = {
            'product_type': 'reanalysis',
            'variable': test_config['variables'],
            'year': [test_config['year']],
            'month': [test_config['month']],
            'day': [test_config['day']],
            'time': [test_config['time']],
            'area': area,
            'format': 'netcdf'
        }
        
        netcdf_target = f'era5_test_netcdf_grid_{grid["id"]}.nc'
        success, size, download_time = download_era5_data(
            test_config['datasets']['netcdf'], netcdf_request, netcdf_target
        )
        
        if success:
            results['netcdf']['success'] += 1
            results['netcdf']['total_size'] += size
            results['netcdf']['total_time'] += download_time
            results['netcdf']['files'].append({
                'grid_id': grid['id'],
                'file': netcdf_target,
                'size_mb': size,
                'time_s': download_time
            })
    
    # Print summary results
    print("\n" + "="*60)
    print("DOWNLOAD EFFICIENCY COMPARISON RESULTS")
    print("="*60)
    
    for format_name, format_results in results.items():
        print(f"\n{format_name.upper()} FORMAT:")
        print(f"  Successful downloads: {format_results['success']}/{len(grids)}")
        if format_results['success'] > 0:
            avg_size = format_results['total_size'] / format_results['success']
            avg_time = format_results['total_time'] / format_results['success']
            print(f"  Average file size: {avg_size:.2f} MB")
            print(f"  Average download time: {avg_time:.2f} seconds")
            print(f"  Total size: {format_results['total_size']:.2f} MB")
            print(f"  Total time: {format_results['total_time']:.2f} seconds")
            print(f"  Average speed: {avg_size/avg_time:.2f} MB/s")
    
    # Efficiency comparison
    if results['grib']['success'] > 0 and results['netcdf']['success'] > 0:
        grib_avg_size = results['grib']['total_size'] / results['grib']['success']
        grib_avg_time = results['grib']['total_time'] / results['grib']['success']
        netcdf_avg_size = results['netcdf']['total_size'] / results['netcdf']['success']
        netcdf_avg_time = results['netcdf']['total_time'] / results['netcdf']['success']
        
        print(f"\nEFFICIENCY COMPARISON:")
        print(f"  Size ratio (GRIB/NetCDF): {grib_avg_size/netcdf_avg_size:.2f}")
        print(f"  Time ratio (GRIB/NetCDF): {grib_avg_time/netcdf_avg_time:.2f}")
        print(f"  Speed ratio (GRIB/NetCDF): {(grib_avg_size/grib_avg_time)/(netcdf_avg_size/netcdf_avg_time):.2f}")
        
        if grib_avg_size < netcdf_avg_size:
            print(f"  GRIB is more space-efficient by {(1-grib_avg_size/netcdf_avg_size)*100:.1f}%")
        else:
            print(f"  NetCDF is more space-efficient by {(1-netcdf_avg_size/grib_avg_size)*100:.1f}%")
            
        if grib_avg_time < netcdf_avg_time:
            print(f"  GRIB is faster by {(1-grib_avg_time/netcdf_avg_time)*100:.1f}%")
        else:
            print(f"  NetCDF is faster by {(1-netcdf_avg_time/grib_avg_time)*100:.1f}%")

if __name__ == "__main__":
    main()